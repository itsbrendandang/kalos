"""Feasibility classifier: producer vs non-producer, to gate acquisition.

Motivation (see BENCHMARK.md): the real media DoE titer is zero-inflated
(~21% non-producers, titer at/near zero). A single GP over that response tries
to fit a spiky feasible/infeasible surface and over-exploits a noisy
incumbent, which is why plain BO loses to random search on that dataset. The
fix here is not a new acquisition function - it is a separate binary
classifier for P(feasible) that gates EI multiplicatively, so the optimizer
spends fewer picks in the non-producer region without changing how EI itself
is computed.

This module only models feasibility. The GP over titer (`kalos.core.surrogate.
Surrogate`) is unchanged and is composed with this module by the caller (see
`kalos/bench/pool.py`'s "bo_feas" / "bo_feas_clean" strategies).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def feasible_labels(y: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Binary feasibility labels from a continuous titer array.

    Assumption: feasible == producer. The default `threshold=0.0` means a row
    is feasible iff `y > threshold` (strictly greater, not `>=`), so an exact
    zero reading is treated as a non-producer. For noisy assays where a
    reported "zero" is actually small positive instrument noise, a small
    positive epsilon threshold may be more appropriate than 0.0 - callers
    working with such assays should pass that epsilon explicitly.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    return (y_arr > threshold).astype(int)


class FeasibilityClassifier:
    """Binary P(feasible) classifier gating acquisition, robust to cold start.

    Wraps `Pipeline(StandardScaler(), LogisticRegression(class_weight=
    "balanced", max_iter=1000))` on binary (0/1) labels.

    Cold-start / degenerate handling (critical): if the labels passed to
    `fit` contain fewer than 2 distinct classes, OR fewer than 3 examples of
    the minority class, sklearn is never fit at all. Instead an internal
    fallback flag is set, and `predict_proba` returns `np.ones(len(X))` -
    meaning "don't gate, defer entirely to EI". This never raises. It is
    exactly what prevents sklearn's single-class fit errors from crashing a
    pool loop early on, before enough non-producers have been observed to
    train a real classifier.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._fallback: bool = True

    def fit(self, X: np.ndarray, y_binary: np.ndarray) -> "FeasibilityClassifier":
        """Fit on binary labels, or fall back to the no-gate state (see class docstring)."""
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y_binary, dtype=int).reshape(-1)
        counts = np.bincount(ya) if ya.size else np.array([])
        n_classes = int((counts > 0).sum())
        minority_count = int(counts[counts > 0].min()) if n_classes else 0
        if n_classes < 2 or minority_count < 3:
            self._fallback = True
            self._pipeline = None
            return self
        self._fallback = False
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ])
        self._pipeline.fit(Xa, ya)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(feasible) as a 1-D float array in [0, 1], length == len(X).

        In the cold-start fallback state (see `fit`), returns `np.ones(len(X))`
        so downstream gating is a no-op rather than an sklearn crash.
        """
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        if self._fallback or self._pipeline is None:
            return np.ones(n, dtype=float)
        proba = self._pipeline.predict_proba(Xa)
        classes = self._pipeline.named_steps["clf"].classes_
        pos_idx = int(np.where(classes == 1)[0][0])
        return np.asarray(proba[:, pos_idx], dtype=float).reshape(-1)


def _classifier_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard 10-bin expected calibration error for a binary classifier.

    This is the CLASSIFIER's ECE - is the stated P(feasible) itself honest -
    not to be confused with `kalos.core.evaluation.interval_calibration`'s
    ECE, which asks whether the surrogate's regression uncertainty bands are
    honest. The two questions are unrelated, but both metrics live on the
    same 0-to-1 "mean absolute gap between nominal and empirical" scale and
    both happen to be named `ece`, which is exactly how they get conflated.

    Predictions are binned into `n_bins` equal-width bins over `[0, 1]` by
    predicted P(feasible) (`y_prob`). Within each non-empty bin, `confidence`
    is the mean predicted probability in that bin and `accuracy` is the
    fraction of rows in that bin that were actually feasible; the bin's
    contribution to ECE is `|confidence - accuracy|` weighted by the bin's
    share of all rows. This is the standard Guo et al. (2017) formulation,
    applied directly to the positive-class probability since feasibility is a
    binary problem - there is no separate "top predicted class" to bin by, as
    there would be for a multi-class classifier.

    Never raises: an empty input returns `nan` (nothing to bin).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    n = len(y_true)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin by the RIGHT edge, closed on the right, so a predicted probability of
    # exactly 1.0 lands in the last bin rather than falling out of every bin.
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (float(mask.sum()) / n) * abs(confidence - accuracy)
    return float(ece)


def feasibility_cv_report(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    threshold: float = 0.0,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Cross-validated feasibility-classifier metrics, for the promotion gate.

    Runs the SAME pooled-OOF CV `feasibility_cv_auc` runs (identical fold
    construction, identical guard rails - `feasibility_cv_auc` is now a thin
    wrapper around this function's `auc` field) and additionally reports
    `brier` and `ece`. These three - AUC, Brier, ECE - are the classifier
    metrics `GatesConfig.min_feasibility_auc`/`max_ece`/`max_brier` were
    originally written against in the lean engine; this is what finally lets
    `check_gates` read real numbers for them instead of blocking on
    "unmeasured" every time.

    `brier` is the mean squared error of the pooled out-of-fold P(feasible)
    against the 0/1 feasibility label - the standard Brier score, lower is
    better, 0.25 is what a coin-flip (constant p=0.5) classifier scores no
    matter the true label mix. `ece` is `_classifier_ece` above: the standard
    10-bin expected calibration error of the SAME pooled OOF predictions.
    Both are computed on the identical `y_true`/`y_pred` pool the AUC uses, so
    all three numbers describe one CV run, not three separately-sampled ones.

    Labels are computed via `feasible_labels(y, threshold)`. If `groups` is
    given, folds are made with `StratifiedGroupKFold` (replicates/campaigns
    never split across train/val); otherwise plain `StratifiedKFold` is used.

    Guard rails (never raises, all carried over from `feasibility_cv_auc`):
    - If the full label set has fewer than 2 classes, every metric is `nan`
      and `evaluable` is `False` (AUC/Brier/ECE are all undefined with a
      single class - a constant P(feasible) cannot be scored against itself).
    - `n_splits` is reduced to at most the minority class count, since a
      stratified split needs at least `n_splits` examples of the minority
      class. If that leaves `n_splits < 2`, every metric is `nan`.
    - When `groups` is given, `n_splits` is ALSO reduced to at most the number
      of distinct groups: sklearn's grouped splitters refuse to make more
      folds than there are groups. The minority-class cap does not imply this
      one - a replicated sheet can hold six non-producing rows across only
      four recipes - and without it this raised `ValueError` on exactly the
      replicated data it exists to score. That mattered because the value
      feeds the fail-closed promotion gate, and an exception is not a closed
      gate; it is a 500 that skips the verdict entirely.
    - Any fold whose test split ends up single-class (AUC undefined for that
      fold) is skipped; if every fold is skipped, or fewer than 2 classes
      remain in the pooled out-of-fold predictions, every metric is `nan`.

    `n`, `n_feasible`, `n_infeasible` describe the POOLED out-of-fold set the
    metrics were actually computed on (not the full input) when `evaluable`
    is `True` - the same rows `auc`/`brier`/`ece` are measured against, since
    a single-class test fold can be skipped and drop rows out of the pool.
    When `evaluable` is `False` no CV ran, so these fall back to the full
    input's label counts, for context on why nothing was measurable.
    """
    Xa = np.asarray(X, dtype=float)
    labels = feasible_labels(y, threshold)
    counts = np.bincount(labels) if labels.size else np.array([])
    n_classes = int((counts > 0).sum())

    def _unavailable() -> dict:
        n_feasible = int(counts[1]) if len(counts) > 1 else 0
        n_infeasible = int(counts[0]) if len(counts) > 0 else 0
        return {
            "auc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "n": int(labels.size),
            "n_feasible": n_feasible,
            "n_infeasible": n_infeasible,
            "evaluable": False,
        }

    if n_classes < 2:
        return _unavailable()

    minority_count = int(counts[counts > 0].min())
    splits = min(n_splits, minority_count)
    g = None if groups is None else np.asarray(groups)
    if g is not None:
        splits = min(splits, int(len(np.unique(g))))
    if splits < 2:
        return _unavailable()

    if g is not None:
        cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
        split_iter = cv.split(Xa, labels, groups=g)
    else:
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        split_iter = cv.split(Xa, labels)

    oof_true: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []
    for train_idx, test_idx in split_iter:
        y_test = labels[test_idx]
        if len(np.unique(y_test)) < 2:
            # AUC is undefined for a single-class test fold; skip its contribution.
            continue
        clf = FeasibilityClassifier().fit(Xa[train_idx], labels[train_idx])
        p = clf.predict_proba(Xa[test_idx])
        oof_true.append(y_test)
        oof_pred.append(p)

    if not oof_true:
        return _unavailable()
    y_true = np.concatenate(oof_true)
    y_pred = np.concatenate(oof_pred)
    if len(np.unique(y_true)) < 2:
        return _unavailable()

    auc = float(roc_auc_score(y_true, y_pred))
    brier = float(np.mean((y_pred - y_true) ** 2))
    ece = _classifier_ece(y_true, y_pred)
    return {
        "auc": auc,
        "brier": brier,
        "ece": ece,
        "n": int(len(y_true)),
        "n_feasible": int((y_true == 1).sum()),
        "n_infeasible": int((y_true == 0).sum()),
        "evaluable": True,
    }


def feasibility_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    threshold: float = 0.0,
    n_splits: int = 5,
    seed: int = 0,
) -> float:
    """Cross-validated AUC of the feasibility classifier, for the promotion gate.

    A thin wrapper around `feasibility_cv_report`'s `auc` field, kept as its
    own function so existing callers (and the tests that pin its exact
    fold-construction and guard-rail behavior) do not have to change. See
    `feasibility_cv_report` for the full contract; this delegates to it rather
    than duplicating the CV loop, so the two can never drift apart.
    """
    return feasibility_cv_report(
        X, y, groups=groups, threshold=threshold, n_splits=n_splits, seed=seed
    )["auc"]


__all__ = [
    "FeasibilityClassifier",
    "feasible_labels",
    "feasibility_cv_auc",
    "feasibility_cv_report",
]

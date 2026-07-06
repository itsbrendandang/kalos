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

    Labels are computed via `feasible_labels(y, threshold)`. If `groups` is
    given, folds are made with `StratifiedGroupKFold` (replicates/campaigns
    never split across train/val); otherwise plain `StratifiedKFold` is used.

    AUC is computed once on the pooled out-of-fold predictions (rather than
    averaged per-fold), which is simpler and standard for CV-AUC reporting
    and avoids weighting small folds equally with large ones.

    Guard rails (never raises):
    - If the full label set has fewer than 2 classes, returns `nan` (AUC is
      undefined with a single class).
    - `n_splits` is reduced to at most the minority class count, since a
      stratified split needs at least `n_splits` examples of the minority
      class. If that leaves `n_splits < 2`, returns `nan`.
    - Any fold whose test split ends up single-class (AUC undefined for that
      fold) is skipped; if every fold is skipped, or fewer than 2 classes
      remain in the pooled out-of-fold predictions, returns `nan`.
    """
    Xa = np.asarray(X, dtype=float)
    labels = feasible_labels(y, threshold)
    counts = np.bincount(labels) if labels.size else np.array([])
    n_classes = int((counts > 0).sum())
    if n_classes < 2:
        return float("nan")

    minority_count = int(counts[counts > 0].min())
    splits = min(n_splits, minority_count)
    if splits < 2:
        return float("nan")

    if groups is not None:
        g = np.asarray(groups)
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
        return float("nan")
    y_true = np.concatenate(oof_true)
    y_pred = np.concatenate(oof_pred)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


__all__ = ["FeasibilityClassifier", "feasible_labels", "feasibility_cv_auc"]

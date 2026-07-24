"""Acquisition: turn scored candidates into the next batch to run.

`predict` scores a cohort; `propose` ranks those candidates by an acquisition
score that balances exploit (predicted titer) against explore (uncertainty), then
picks a diversified batch. The strategy is gated on the honest verdict:

- at least one model validated (some `weight > 0`) -> EXPLOIT: rank by an
  upper-confidence bound `mean + beta * uncertainty` and climb toward the optimum;
- nothing validated -> EXPLORE: the predicted titers are not trustworthy, so rank
  by uncertainty alone (space-filling) to gather data that can validate the model.

Uncertainty combines each model's own predicted std (aleatoric) with cross-model
disagreement (epistemic). This is a deliberately simple, dependency-free stand-in
for a real Bayesian-optimization acquisition (kalos / BoTorch qEI/qUCB); see TODO.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProposalResult:
    table: pd.DataFrame          # per-candidate acq scores, ranked, with a `selected` flag
    mode: str                    # "exploit" | "explore"
    q: int
    beta: float
    validated_models: list[str]
    note: str


def _ensemble(results) -> tuple[np.ndarray, np.ndarray]:
    """Weighted ensemble mean + total uncertainty over `results`. Weight is each
    method's blend weight (0 for unvalidated/collapsed); when every weight is 0
    (explore mode) the mean falls back to an equal-weight estimate. Uncertainty =
    aleatoric (each model's std) combined with epistemic (cross-model disagreement)."""
    means = np.array([r.client_mean for r in results], dtype=float)   # [n_models, n_cand]
    stds = np.array([r.client_std for r in results], dtype=float)
    w = np.array([max(r.weight, 0.0) for r in results], dtype=float)
    w = w / w.sum() if w.sum() > 1e-9 else np.full(len(results), 1.0 / len(results))
    mean = (w[:, None] * means).sum(0)
    aleatoric = (w[:, None] * stds).sum(0)
    disagreement = np.sqrt((w[:, None] * (means - mean) ** 2).sum(0))
    return mean, np.sqrt(aleatoric ** 2 + disagreement ** 2)


def _diversified(feats: np.ndarray | None, q: int, min_dist: float) -> list[int]:
    """Greedy pick down candidates already sorted acq-desc (index order), keeping a
    candidate only if it is at least `min_dist` (standardized) from every one already
    picked, then topping up by acq order if diversity can't fill q."""
    picked: list[int] = []
    n = len(feats) if feats is not None else 0
    if feats is not None and n:
        z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
        for i in range(n):
            if all(np.linalg.norm(z[i] - z[j]) > min_dist for j in picked):
                picked.append(i)
            if len(picked) == q:
                break
    for i in range(n):                       # top up (and the no-feature fallback)
        if len(picked) >= q:
            break
        if i not in picked:
            picked.append(i)
    return picked[:q]


def propose_batch(report, *, q: int = 5, beta: float = 1.5,
                  diversity: float = 1.5) -> ProposalResult:
    """Rank the scored candidates in `report` and select the next batch of `q`."""
    results = report.results
    validated = [r for r in results if r.weight > 0]
    mode, used = ("exploit", validated) if validated else ("explore", results)

    mean, unc = _ensemble(used)
    acq = mean + beta * unc if mode == "exploit" else unc

    table = pd.DataFrame({
        "well_id": results[0].client_well_ids,
        "pred_titer": np.round(mean, 3),
        "uncertainty": np.round(unc, 3),
        "acq_score": np.round(acq, 3),
    })
    table["acq_rank"] = table["acq_score"].rank(ascending=False, method="first").astype(int)
    table = table.sort_values("acq_rank").reset_index(drop=True)

    # Diversity metric over the reference-selected features (fall back to every
    # numeric cohort column), aligned to the acq-sorted candidate order.
    feats = None
    cohort = report.cohort
    if cohort is not None:
        cols = (report.reference or {}).get("feature_cols") or [
            c for c in cohort.columns
            if c != "well_id" and pd.api.types.is_numeric_dtype(cohort[c])]
        cols = [c for c in cols if c in cohort.columns]
        if cols:
            feats = (cohort.set_index("well_id")
                     .loc[table["well_id"], cols].to_numpy(dtype=float))

    picked = _diversified(feats, min(q, len(table)), diversity)
    table["selected"] = False
    table.loc[picked, "selected"] = True
    table["mode"] = mode

    note = (f"UCB = mean + {beta}*uncertainty over validated models "
            f"{[r.name for r in validated]}; batch diversified >{diversity} sd apart."
            if mode == "exploit" else
            "no model validated, so predicted titers are NOT trustworthy; ranking by "
            "uncertainty for space-filling to gather data that can validate the model.")
    return ProposalResult(table=table, mode=mode, q=q, beta=beta,
                          validated_models=[r.name for r in validated], note=note)

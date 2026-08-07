"""Combine multiple MethodResults into one ranked recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import MethodResult


class Blender:
    def __init__(self, results: list[MethodResult]):
        if not results:
            raise ValueError("Blender needs at least one MethodResult")
        self.results = results

    def weights(self) -> dict[str, float]:
        raw = {r.name: r.weight for r in self.results}
        total = sum(raw.values())
        if total <= 1e-9:
            return {k: 1.0 / len(raw) for k in raw}  # equal-weight consensus fallback
        return {k: v / total for k, v in raw.items()}

    def note(self) -> str:
        total = sum(r.weight for r in self.results)
        return ("No method individually validated -- equal-weight consensus fallback."
                if total <= 1e-9 else "Weighted by CV Spearman among methods whose "
                "bootstrap CI excludes 0 (validated); unvalidated or collapsed methods "
                "are zeroed out.")

    def blended_table(self) -> pd.DataFrame:
        well_ids = self.results[0].client_well_ids
        w = self.weights()
        score = np.zeros(len(well_ids), dtype=float)
        for r in self.results:
            score += w[r.name] * r.client_mean

        table = pd.DataFrame({"well_id": well_ids, "blended_score": np.round(score, 4)})
        table["blend_rank"] = table["blended_score"].rank(ascending=False, method="min").astype(int)

        # Cross-model agreement counts only methods that earned weight -- i.e.
        # validated (bootstrap CI excludes 0) and not collapsed. An unvalidated
        # method agreeing on the top wells must not inflate the confidence tier.
        contributing = [r for r in self.results if r.weight > 0]
        top10_sets = [set(pd.Series(r.client_well_ids)[np.argsort(-r.client_mean)[:10]])
                      for r in contributing]
        table["n_methods_agreeing_top10"] = table["well_id"].apply(
            lambda w_id: sum(w_id in s for s in top10_sets))
        table["confidence_tier"] = pd.cut(table["n_methods_agreeing_top10"],
                                          bins=[-1, 1, 3, 99], labels=["low", "medium", "high"])
        return table.sort_values("blend_rank").reset_index(drop=True)

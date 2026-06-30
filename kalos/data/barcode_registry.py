"""Barcode registry — the organized, queryable home for all run data.

Every dataset gets a dataset barcode; every run gets a stable run barcode. Runs
are anonymized on registration (raw client/strain identity never stored). The
registry is the single source of truth: look a run up by barcode in O(1), filter
by dataset, export a training DataFrame, persist to one JSON file, and read a
per-dataset manifest. This is how "all the data" gets organized.

    register_dataset(name, df)  ->  dataset barcode + a barcode per row
    get(barcode)                ->  the run record
    filter(dataset_id=...)      ->  matching barcodes
    to_dataframe(barcodes)      ->  flat features+results table for training
    manifest()                  ->  one row per dataset (counts, columns, prefix)
    save(path) / load(path)     ->  JSON round-trip
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .anonymizer import Anonymizer


@dataclass
class RunRecord:
    barcode: str
    dataset_id: str
    dataset_barcode: str
    features: Dict[str, float]
    results: Dict[str, float]
    meta: Dict


class BarcodeRegistry:
    def __init__(self, salt: str = "kalos") -> None:
        self.anon = Anonymizer(salt=salt)
        self.runs: Dict[str, RunRecord] = {}
        self.datasets: Dict[str, dict] = {}

    # -- registration -------------------------------------------------------- #
    def register_run(
        self, dataset_id: str, dataset_barcode: str, features: Dict, results: Dict, meta: Optional[Dict] = None
    ) -> str:
        barcode = self.anon.run_barcode(dataset_id, features, results)
        if barcode not in self.runs:  # idempotent: same content -> same barcode
            self.runs[barcode] = RunRecord(
                barcode=barcode, dataset_id=dataset_id, dataset_barcode=dataset_barcode,
                features=dict(features), results=dict(results), meta=self.anon.anonymize_meta(meta),
            )
        return barcode

    def register_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        result_cols: Optional[Sequence[str]] = None,
        meta_cols: Optional[Sequence[str]] = None,
    ) -> dict:
        """Register a run sheet. result_cols = the measured outputs (titer, purity,
        ...); everything else numeric is a feature; meta_cols are kept (anonymized)
        as metadata. Returns the dataset barcode and the per-row barcodes."""
        dsbar = self.anon.dataset_barcode(name)
        result_cols = list(result_cols or [])
        meta_cols = list(meta_cols or [])
        feat_cols = [
            c for c in df.columns
            if c not in result_cols and c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        barcodes: List[str] = []
        for _, row in df.iterrows():
            feats = {c: _num(row[c]) for c in feat_cols}
            res = {c: _num(row[c]) for c in result_cols}
            meta = {c: (None if pd.isna(row[c]) else row[c]) for c in meta_cols}
            barcodes.append(self.register_run(name, dsbar, feats, res, meta))
        self.datasets[name] = {
            "dataset_id": name, "dataset_barcode": dsbar, "n_runs": len(barcodes),
            "feature_cols": feat_cols, "result_cols": result_cols, "meta_cols": meta_cols,
            "barcodes": barcodes,
        }
        return self.datasets[name]

    # -- query --------------------------------------------------------------- #
    def get(self, barcode: str) -> RunRecord:
        return self.runs[barcode]

    def filter(self, dataset_id: Optional[str] = None) -> List[str]:
        return [b for b, r in self.runs.items() if dataset_id is None or r.dataset_id == dataset_id]

    def to_dataframe(self, barcodes: Optional[Sequence[str]] = None) -> pd.DataFrame:
        bs = list(barcodes if barcodes is not None else self.runs.keys())
        rows = []
        for b in bs:
            r = self.runs[b]
            rows.append({"barcode": b, "dataset_id": r.dataset_id, **r.features, **r.results})
        return pd.DataFrame(rows)

    def manifest(self) -> List[dict]:
        return [
            {k: v for k, v in d.items() if k != "barcodes"} | {"barcode_sample": d["barcodes"][:3]}
            for d in self.datasets.values()
        ]

    # -- persistence --------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "datasets": self.datasets,
            "runs": {b: asdict(r) for b, r in self.runs.items()},
        }, indent=2, default=_jsonable))

    @classmethod
    def load(cls, path: str | Path) -> "BarcodeRegistry":
        d = json.loads(Path(path).read_text())
        reg = cls()
        reg.datasets = d.get("datasets", {})
        reg.runs = {b: RunRecord(**r) for b, r in d.get("runs", {}).items()}
        return reg

    def __len__(self) -> int:
        return len(self.runs)


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


__all__ = ["BarcodeRegistry", "RunRecord"]

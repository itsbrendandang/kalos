"""Ingestion feed — the platform pushes machine data in; the engine reads it.

A decoupled push model (ported from the lean engine's redesign): an instrument
computer streams runs into a `DataFeed` (a CSV or a folder of CSVs); Kalos reads
the latest feed whenever it proposes the next batch and writes proposals to a
`ProposalSink`. The feed anonymizes on read, so raw client/strain identity never
reaches the model. Both contracts are swappable for a DB/HTTP backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from kalos.data.anonymizer import _hash

# Conservative identity scrubbing on the column level (the Anonymizer handles
# record meta; this handles a wide run sheet). Exact names + a few safe substrings.
_DROP_EXACT = {
    "client_id", "client", "customer", "customer_id", "name", "operator",
    "patient", "patient_id", "sample_id", "sampleid", "strain", "strain_id",
    "donor", "subject", "email", "phone", "mrn", "run_by",
}
_DROP_SUBSTR = ("client", "strain", "operator", "patient", "donor", "email", "phone")
_HASH_EXACT = {"campaign_id", "campaign"}


def anonymize_frame(df: pd.DataFrame, salt: str = "kalos") -> Tuple[pd.DataFrame, List[str]]:
    """Drop identity columns, hash campaign ids. Returns (clean_df, dropped_cols)."""
    dropped: List[str] = []
    out = df.copy()
    for c in list(out.columns):
        lc = str(c).strip().lower()
        if lc in _DROP_EXACT or any(tok in lc for tok in _DROP_SUBSTR):
            out = out.drop(columns=[c])
            dropped.append(str(c))
        elif lc in _HASH_EXACT:
            out[c] = out[c].map(lambda v: _hash(v, salt))
    return out, sorted(dropped)


class DataFeed(ABC):
    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Return every measured run so far (already anonymized)."""


class ProposalSink(ABC):
    @abstractmethod
    def emit(self, recipes: pd.DataFrame, round_id: str) -> Path:
        """Persist a proposed batch; return where it was written."""


class FileDataFeed(DataFeed):
    """Read a CSV/TSV file, or concatenate every *.csv in a folder (append-only)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _raw(self) -> pd.DataFrame:
        if self.path.is_dir():
            files = sorted(self.path.glob("*.csv"))
            return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
        if self.path.exists():
            sep = "\t" if self.path.suffix.lower() == ".tsv" else ","
            return pd.read_csv(self.path, sep=sep)
        return pd.DataFrame()

    def read(self) -> pd.DataFrame:
        return anonymize_frame(self._raw())[0]


class FileProposalSink(ProposalSink):
    """Write each proposed batch to out_dir/proposals_<round_id>.csv for the lab."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)

    def emit(self, recipes: pd.DataFrame, round_id: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"proposals_{round_id}.csv"
        recipes.to_csv(path, index=False)
        return path


__all__ = ["anonymize_frame", "DataFeed", "ProposalSink", "FileDataFeed", "FileProposalSink"]

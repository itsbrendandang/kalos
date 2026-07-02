"""Anonymizer + stable barcode minting.

Ported from the lean engine's data_moat/anonymize.py. Two jobs:
  1. Strip client identity from a run's metadata before it enters the registry
     (drop client/strain/operator, hash campaign/lot to an irreversible pseudonym).
  2. Mint a stable BARCODE for a run or dataset — a short, content-derived id so
     the same run always gets the same barcode (idempotent registration) and no
     raw identity is needed to reference it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def _hash(value: object, salt: str = "kalos") -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


def _stable_payload(parts: Dict) -> str:
    # json with sorted keys -> deterministic regardless of dict order / float repr
    return json.dumps(parts, sort_keys=True, default=str)


@dataclass
class Anonymizer:
    # metadata keys that identify a client/sample and are dropped entirely
    drop_meta_keys: Tuple[str, ...] = (
        "client_id", "client", "customer", "name", "operator", "patient",
        "sample_id", "sampleid", "strain", "donor", "subject", "email", "phone", "mrn",
    )
    # meta keys kept but as a stable irreversible hash (e.g. a campaign/lot id for grouping)
    hash_meta_keys: Tuple[str, ...] = ("campaign_id", "campaign", "lot", "batch_id")
    salt: str = "kalos"
    # substring matches make scrubbing robust to naming ("Sample Name", "Experiment ID", ...)
    _drop_substr: Tuple[str, ...] = ("client", "customer", "strain", "operator", "patient",
                                     "donor", "subject", "email", "phone", "mrn", "sample")
    _hash_substr: Tuple[str, ...] = ("campaign", "experiment", "lot", "batch")

    def anonymize_meta(self, meta: Optional[Dict] = None) -> Dict:
        clean: Dict = {}
        for k, v in dict(meta or {}).items():
            lk = str(k).strip().lower()
            if lk in self.drop_meta_keys or any(t in lk for t in self._drop_substr):
                continue  # identity dropped entirely
            if lk in self.hash_meta_keys or any(t in lk for t in self._hash_substr):
                clean[k] = _hash(v, self.salt)  # pseudonymous, irreversible
            else:
                clean[k] = v
        clean["anonymized"] = True
        return clean

    def run_barcode(self, dataset_id: str, features: Dict, results: Dict, prefix: str = "KAL") -> str:
        """Content-stable barcode for one run: same recipe+result -> same barcode."""
        h = _hash(_stable_payload({"d": dataset_id, "f": features, "r": results}), self.salt)
        return f"{prefix}-{h[:16].upper()}"

    def dataset_barcode(self, name: str, prefix: str = "KAL-DS") -> str:
        return f"{prefix}-{_hash(name, self.salt)[:6].upper()}"


__all__ = ["Anonymizer", "_hash"]

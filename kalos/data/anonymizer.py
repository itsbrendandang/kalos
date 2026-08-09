"""Strip client identity from run metadata.

Ported from the lean engine's data_moat/anonymize.py. One job: drop the columns
that name a client (client/strain/operator) and replace grouping ids
(campaign/lot/batch) with an irreversible salted hash, so a run can still be
grouped for leakage-safe CV without carrying who it belongs to.

The barcode-minting half of this module was removed along with
`kalos.data.barcode_registry`: barcoding is explicitly off the roadmap, and the
minting helpers had no caller left once the registry went.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

log = logging.getLogger("kalos.anonymizer")

# --- salt: read from the environment, never a silent hardcoded default -------- #
# The barcode / identity hash is only pseudonymous, not truly anonymous: an
# attacker who knows the salt can dictionary-attack short identifiers (a strain
# name, a sample id) back out of a hash. In production the salt MUST come from the
# environment. We keep a literal DEV fallback so local runs and tests still work,
# but emit a LOUD warning when it is used, because shipping with it makes every
# identity hash guessable. Barcodes must stay STABLE across runs (idempotent
# registration), so the fallback is a fixed literal, not a random per-process value.
_ENV_SALT_VAR = "KALOS_ANON_SALT"
_DEV_SALT = "kalos"
_warned_default_salt = False


def default_salt() -> str:
    """Return the anonymization salt: `KALOS_ANON_SALT` if set, else the dev salt.

    When the env var is unset we fall back to the literal dev salt so local use
    keeps working, but log a one-time warning that production MUST set it, since
    identity hashes are otherwise dictionary-attackable.
    """
    salt = os.environ.get(_ENV_SALT_VAR)
    if salt:
        return salt
    global _warned_default_salt
    if not _warned_default_salt:
        log.warning(
            "%s is not set; falling back to the built-in dev salt. Identity "
            "hashes and barcodes are GUESSABLE with it - production MUST set %s.",
            _ENV_SALT_VAR,
            _ENV_SALT_VAR,
        )
        _warned_default_salt = True
    return _DEV_SALT


def _hash(value: object, salt: str | None = None) -> str:
    if salt is None:
        salt = default_salt()
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


# --- canonical identity-scrub rules (single source of truth) ------------------ #
# One place defines which columns/keys are identity and get DROPPED, and which are
# grouping ids that get HASHED to a stable pseudonym. `ingest/feed.py` imports
# these instead of maintaining a second, drift-prone copy. The lists are the UNION
# of the two rule-sets that previously lived here and in `feed.py`, so nothing
# that was being scrubbed by either path stops being scrubbed.
DROP_EXACT: frozenset[str] = frozenset({
    "client_id", "client", "customer", "customer_id", "name", "operator",
    "patient", "patient_id", "sample_id", "sampleid", "strain", "strain_id",
    "donor", "subject", "email", "phone", "mrn", "run_by",
})
# substring matches make scrubbing robust to naming ("Sample Name", "Experiment ID")
DROP_SUBSTR: tuple[str, ...] = (
    "client", "customer", "strain", "operator", "patient", "donor", "subject",
    "email", "phone", "mrn", "sample",
)
# grouping ids kept but replaced with a stable irreversible hash (campaign/lot/batch)
HASH_EXACT: frozenset[str] = frozenset({"campaign_id", "campaign", "lot", "batch_id"})
HASH_SUBSTR: tuple[str, ...] = ("campaign", "experiment", "lot", "batch")


@dataclass
class Anonymizer:
    # All scrub rules come from the canonical lists above (single source of truth,
    # shared with ingest/feed.py). `salt` defaults to the env-configured salt.
    drop_meta_keys: frozenset[str] = DROP_EXACT
    hash_meta_keys: frozenset[str] = HASH_EXACT
    salt: Optional[str] = None
    _drop_substr: Tuple[str, ...] = DROP_SUBSTR
    _hash_substr: Tuple[str, ...] = HASH_SUBSTR

    def _salt(self) -> str:
        return self.salt if self.salt is not None else default_salt()

    def anonymize_meta(self, meta: Optional[Dict] = None) -> Dict:
        clean: Dict = {}
        salt = self._salt()
        for k, v in dict(meta or {}).items():
            lk = str(k).strip().lower()
            if lk in self.drop_meta_keys or any(t in lk for t in self._drop_substr):
                continue  # identity dropped entirely
            if lk in self.hash_meta_keys or any(t in lk for t in self._hash_substr):
                clean[k] = _hash(v, salt)  # pseudonymous, irreversible
            else:
                clean[k] = v
        clean["anonymized"] = True
        return clean


__all__ = [
    "Anonymizer",
    "_hash",
    "default_salt",
    "DROP_EXACT",
    "DROP_SUBSTR",
    "HASH_EXACT",
    "HASH_SUBSTR",
]

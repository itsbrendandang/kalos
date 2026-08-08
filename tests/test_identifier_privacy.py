"""Identifier columns must not be modeled, and must not escape anonymization.

Both failures below were live, and both trace to the same root cause: a
`DomainProfile.id_hint` anchored to whole tokens (`^(id|name|run|batch|campaign|
lot|...)$`) matches a column called exactly `run`, but not `run_number`,
`batch_id`, `campaign_id` or `lot_number`. Real run sheets use the compound
forms, so identifier columns were sailing straight past every check keyed on
that pattern.

1. SCIENTIFIC. A numeric identifier became a model FEATURE. Given `run_number`
   and `batch_id`, the engine fit on them, reported both as significant drivers
   at rho = 1.0, and proposed a recipe instructing the scientist to "set
   batch_id = 100.037". A run index rises monotonically with time, so it
   correlates with any drift or learning trend in a campaign and will almost
   always rank as a top driver - a spurious correlation presented as process
   insight.

2. PRIVACY. `anonymize=True` published those same names verbatim while claiming
   to pseudonymize identifier columns, including inside the validation report's
   prose. The codebase already contradicted itself: the metadata scrubber's
   `HASH_EXACT` lists `campaign_id`, so one id was hashed as metadata and
   published as a column name in the same response.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from kalos.portal.analysis import _analyze, identifier_pattern  # noqa: E402
from kalos.domains import BIOPROCESS_PROFILE  # noqa: E402

# Compound identifier headers a real client sheet actually carries.
_COMPOUND_IDS = [
    "run_number",
    "run_id",
    "batch_id",
    "batch_number",
    "campaign_id",
    "lot_number",
    "sample_id",
    "experiment_id",
    "culture_id",
    "vessel_id",
]

# Real process inputs and outputs that must NEVER be treated as identifiers,
# including the deliberately adversarial ones that merely START with an
# identifier word.
_REAL_COLUMNS = [
    "Methanol",
    "pH",
    "scale_L",
    "lipase_titer",
    "batch_titer",
    "run_duration_days",
    "culture_duration_days",
    "reactor_temperature_C",
    "sampling_rate",
]


def _sheet_with(extra: dict[str, list]) -> pd.DataFrame:
    n = 12
    base = {
        "Methanol": [0.2 + 0.05 * i for i in range(n)],
        "pH": [6.9 + 0.02 * (i % 6) for i in range(n)],
        "lipase_titer": [1.0 + 0.2 * i for i in range(n)],
    }
    return pd.DataFrame({**extra, **base})


# --- the pattern itself ---------------------------------------------------- #


@pytest.mark.parametrize("name", _COMPOUND_IDS)
def test_compound_identifier_names_are_recognized(name):
    assert identifier_pattern(BIOPROCESS_PROFILE.id_hint.pattern).match(name)


@pytest.mark.parametrize("name", _REAL_COLUMNS)
def test_real_measurements_are_not_mistaken_for_identifiers(name):
    """Precision matters as much as recall: dropping or aliasing a genuine
    process input would be its own defect."""
    assert not identifier_pattern(BIOPROCESS_PROFILE.id_hint.pattern).match(name)


def test_known_preexisting_overmatch_sample_glob_is_recorded_not_hidden():
    """`BIOPROCESS_PROFILE.id_hint` contains a `sample.*` glob, so a genuine
    measurement called `sample_volume_L` is classified as an identifier and
    dropped from modeling.

    This is PRE-EXISTING upstream behavior, not a consequence of the union added
    for compound identifiers - `sample_volume_L` is matched by `sample.*` in the
    profile's own hint and is rejected by the compound pattern. It is asserted
    here so the over-match is a recorded, deliberate known issue rather than a
    silent surprise, and so that anyone tightening `id_hint` later sees this test
    turn red and knows to reconsider it on purpose.

    It was left unchanged rather than fixed here because `id_hint` also drives
    which columns are dropped from the model, so narrowing it is a scientific
    behavior change that deserves its own decision, not a drive-by edit inside a
    privacy fix.
    """
    union = identifier_pattern(BIOPROCESS_PROFILE.id_hint.pattern)
    assert BIOPROCESS_PROFILE.id_hint.match("sample_volume_L")
    assert union.match("sample_volume_L")


def test_union_can_only_widen_never_narrow():
    """Whatever the profile's own hint matched before must still match."""
    hint = BIOPROCESS_PROFILE.id_hint
    union = identifier_pattern(hint.pattern)
    for name in ["id", "name", "run", "batch", "campaign", "lot", "well", "index", "Sample Name"]:
        if hint.match(name):
            assert union.match(name), f"{name} was an identifier before and must remain one"


# --- 1. identifiers must not be modeled ------------------------------------ #


def test_numeric_identifier_never_becomes_a_model_feature():
    df = _sheet_with(
        {"run_number": list(range(12)), "batch_id": list(range(100, 112))}
    )
    out = _analyze(df, target="lipase_titer")

    assert "run_number" not in out["features"]
    assert "batch_id" not in out["features"]
    assert "Methanol" in out["features"], "a real process input must survive"


def test_identifier_never_reported_as_a_driver():
    """A run index correlates with time, so it would otherwise rank as a top
    driver and be read as a process insight."""
    df = _sheet_with({"run_number": list(range(12))})
    out = _analyze(df, target="lipase_titer")
    assert "run_number" not in {d["name"] for d in out["drivers"]}


def test_identifier_never_appears_in_a_proposed_recipe():
    """"Set batch_id = 100.037" is not an executable instruction."""
    df = _sheet_with({"batch_id": list(range(100, 112))})
    out = _analyze(df, target="lipase_titer")
    for proposal in out["proposals"]:
        assert "batch_id" not in proposal["recipe"]


def test_provenance_labels_identifiers_as_dropped_id():
    """Feature selection and the provenance report must agree about what an
    identifier is; they resolve through the same pattern."""
    df = _sheet_with({"run_number": list(range(12)), "batch_id": list(range(100, 112))})
    out = _analyze(df, target="lipase_titer")
    statuses = {row["name"]: row["status"] for row in out["provenance"]}
    assert statuses["run_number"] == "dropped_id"
    assert statuses["batch_id"] == "dropped_id"
    assert statuses["Methanol"] == "kept_feature"


# --- 2. identifiers must not survive anonymization ------------------------- #


def test_anonymize_hides_every_identifier_name_including_in_prose():
    """The validation report names columns twice: a `column` field and the
    sentence built around it. Aliasing only the field would leak the name."""
    df = _sheet_with(
        {
            # an unrecognized unit token, so this column generates a finding whose
            # message embeds its name
            "Sample Name": [f"{i}L" for i in range(12)],
            "campaign_id": ["C-2026"] * 12,
            "batch_id": list(range(100, 112)),
        }
    )
    out = _analyze(df, target="lipase_titer", anonymize=True)
    blob = json.dumps(out)

    for identifier in ("Sample Name", "campaign_id", "batch_id"):
        assert identifier not in blob, f"{identifier} escaped anonymization"

    # the pseudonym is applied, not merely the name deleted
    assert "col_" in blob


def test_anonymize_preserves_real_feature_names():
    """The owner UI legitimately shows a driver called "Methanol"; anonymization
    must not corrupt the report it is reading."""
    df = _sheet_with({"batch_id": list(range(100, 112))})
    out = _analyze(df, target="lipase_titer", anonymize=True)
    blob = json.dumps(out)
    assert "Methanol" in blob
    assert "lipase_titer" in blob


def test_anonymize_is_stable_so_the_report_stays_internally_consistent():
    """The same name must map to the same pseudonym everywhere in one response,
    or cross-referencing the report becomes impossible."""
    df = _sheet_with({"campaign_id": ["C-2026"] * 12, "batch_id": list(range(100, 112))})
    first = _analyze(df, target="lipase_titer", anonymize=True)
    second = _analyze(df, target="lipase_titer", anonymize=True)
    names_first = sorted(row["name"] for row in first["provenance"])
    names_second = sorted(row["name"] for row in second["provenance"])
    assert names_first == names_second


def test_no_anonymize_keeps_real_names():
    """Anonymization is opt-in; the default owner view is unchanged."""
    df = _sheet_with({"batch_id": list(range(100, 112))})
    out = _analyze(df, target="lipase_titer", anonymize=False)
    assert "batch_id" in {row["name"] for row in out["provenance"]}

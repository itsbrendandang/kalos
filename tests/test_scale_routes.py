"""Tests for `kalos.portal.scale_routes` (`POST /api/scale/readout`), through
FastAPI's `TestClient`. Follows the auth-setup pattern in
`tests/test_auth.py`/`tests/test_tenant_isolation.py` (env-configured tokens
for 401/403; open mode otherwise) and the store-override pattern in
`tests/test_healthz.py` (a `tmp_path` `CampaignStore`, never the real
`~/.kalos/portal.db`).
"""
from __future__ import annotations

import hashlib
import json
import re

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app, get_campaign_store  # noqa: E402
from kalos.portal.auth import READ, WRITE  # noqa: E402
from kalos.portal.campaign import CampaignStore  # noqa: E402
from kalos.portal.scale_routes import INTENDED_USE_STATEMENT  # noqa: E402

_DEMO_CSV = "examples/synthetic_scaleup/synthetic_scaleup.csv"


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _configure_tokens(monkeypatch, records: list[dict]) -> None:
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(records))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)


@pytest.fixture
def client(tmp_path):
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _demo_bytes() -> bytes:
    with open(_DEMO_CSV, "rb") as f:
        return f.read()


def _target_json(**overrides) -> str:
    target = {
        "scale_L": 7500.0,  # 1.5x the largest trained scale (5000 L)
        "agitation_rpm": 150.0,
        "airflow_L_per_min": 300.0,
        "target_column": "titer_g_per_L",
        "process_params": {"ph_setpoint": 7.2, "temperature_C": 37.0},
    }
    target.update(overrides)
    return json.dumps(target)


def _post(tc, *, target_json: str, csv_bytes: bytes | None = None, headers: dict | None = None):
    return tc.post(
        "/api/scale/readout",
        data={"target": target_json},
        files={"file": ("run_sheet.csv", csv_bytes if csv_bytes is not None else _demo_bytes(), "text/csv")},
        headers=headers or {},
    )


# --------------------------------------------------------------------------- #
# 200 HTML, every required section present, exact intended-use sentence
# --------------------------------------------------------------------------- #


def test_demo_sheet_1p5x_target_returns_200_html_with_every_section(client):
    tc, _store = client
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    body = r.text

    assert INTENDED_USE_STATEMENT in body
    assert "approximate coverage" in body
    # decision banner
    assert "NUMBER" in body
    # target inputs
    assert "7500" in body
    assert "ph_setpoint" in body
    # per-rung table
    assert "Backtest ladder" in body
    assert "beats both" in body
    # reference vs requested ratio
    assert "Step ratio" in body
    assert "Reference (backtested) ratio" in body
    assert "Requested (target) ratio" in body
    # baseline comparison
    assert "Baseline comparison" in body
    # physics assumptions
    assert "Physics assumptions" in body
    assert "power_number" in body
    # provenance
    assert "Provenance" in body
    assert "kalos git SHA" in body
    assert "raw upload SHA-256" in body
    assert "normalized frame SHA-256" in body


def test_intended_use_statement_is_exact(client):
    tc, _store = client
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 200
    assert (
        "Development decision support only. Not a GMP or regulatory record. "
        "Not validated under 21 CFR Part 11."
    ) in r.text


# --------------------------------------------------------------------------- #
# 422 naming the failed check
# --------------------------------------------------------------------------- #


def test_bad_target_json_is_422_with_failed_check(client):
    tc, _store = client
    r = _post(tc, target_json="not json")
    assert r.status_code == 422
    body = r.json()
    assert body["failed_check"] == "invalid_target"
    assert "detail" in body


def test_gate_failure_is_422_naming_the_check(client):
    tc, _store = client
    # target scale below the largest trained scale -> interpolation refusal
    r = _post(tc, target_json=_target_json(scale_L=100.0))
    assert r.status_code == 422
    body = r.json()
    assert body["failed_check"] == "target_not_interpolation"


def test_ratio_hard_cap_gate_failure_is_422(client):
    tc, _store = client
    r = _post(tc, target_json=_target_json(scale_L=5000.0 * 25))
    assert r.status_code == 422
    assert r.json()["failed_check"] == "target_ratio_hard_cap"


def test_missing_run_sheet_column_is_422(client):
    tc, _store = client
    bad_csv = pd.read_csv(_DEMO_CSV).drop(columns=["ph_setpoint"]).to_csv(index=False).encode()
    r = _post(tc, target_json=_target_json(), csv_bytes=bad_csv)
    assert r.status_code == 422
    assert r.json()["failed_check"] == "required_columns"


# --------------------------------------------------------------------------- #
# scale-only fallback: agitation_rpm/airflow_L_per_min optional (owner
# decision 2026-09-24)
# --------------------------------------------------------------------------- #


def test_dropped_columns_demo_is_200_scale_only_with_the_note_and_not_recorded(client):
    tc, _store = client
    dropped_csv = pd.read_csv(_DEMO_CSV).drop(columns=["agitation_rpm", "airflow_L_per_min"]).to_csv(index=False).encode()
    target_json = json.dumps(
        {
            "scale_L": 7500.0,
            "target_column": "titer_g_per_L",
            "process_params": {"ph_setpoint": 7.2, "temperature_C": 37.0},
        }
    )
    r = _post(tc, target_json=target_json, csv_bytes=dropped_csv)
    assert r.status_code == 200, r.text
    page = r.text
    assert (
        "Scale-only model: this sheet does not record agitation_rpm and airflow_L_per_min, "
        "so mixing and oxygen-transfer effects are not modeled." in page
    )
    assert "not recorded" in page
    assert "note-info" in page


def test_sheet_with_both_columns_but_target_missing_agitation_is_422_invalid_target(client):
    tc, _store = client
    target_json = json.dumps(
        {
            "scale_L": 7500.0,
            "airflow_L_per_min": 300.0,
            "target_column": "titer_g_per_L",
            "process_params": {"ph_setpoint": 7.2, "temperature_C": 37.0},
        }
    )
    r = _post(tc, target_json=target_json)
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["failed_check"] == "invalid_target"
    assert "agitation_rpm" in body["detail"]


# --------------------------------------------------------------------------- #
# 401/403 without the write scope
# --------------------------------------------------------------------------- #


def test_unauthenticated_request_is_401(client, monkeypatch):
    tc, _store = client
    _configure_tokens(monkeypatch, [{"token_sha256": _sha("wtok"), "subject": "s", "tenant": "t", "scopes": [WRITE]}])
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 401


def test_read_only_token_is_403(client, monkeypatch):
    tc, _store = client
    _configure_tokens(monkeypatch, [{"token_sha256": _sha("rtok"), "subject": "s", "tenant": "t", "scopes": [READ]}])
    r = _post(tc, target_json=_target_json(), headers={"Authorization": "Bearer rtok"})
    assert r.status_code == 403


def test_valid_write_token_succeeds(client, monkeypatch):
    tc, _store = client
    _configure_tokens(monkeypatch, [{"token_sha256": _sha("wtok"), "subject": "s", "tenant": "t", "scopes": [WRITE]}])
    r = _post(tc, target_json=_target_json(), headers={"Authorization": "Bearer wtok"})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# stateless: nothing written to disk/DB
# --------------------------------------------------------------------------- #


def test_readout_writes_nothing_to_the_campaign_store(client):
    tc, store = client
    assert store.get() is None
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 200
    assert store.get() is None


def test_readout_creates_no_new_files_on_disk(client, tmp_path):
    tc, _store = client
    before = set(tmp_path.iterdir())
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 200
    after = set(tmp_path.iterdir())
    assert after == before


# --------------------------------------------------------------------------- #
# one page, number formatting, human decision banners, timestamp
# --------------------------------------------------------------------------- #


_DECIMAL_RE = re.compile(r"\d+\.(\d+)")


def test_no_rendered_number_has_more_than_4_decimal_places(client):
    tc, _store = client
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 200
    offenders = [m.group(0) for m in _DECIMAL_RE.finditer(r.text) if len(m.group(1)) > 4]
    assert offenders == []


def test_generated_at_timestamp_in_header(client):
    tc, _store = client
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 200
    assert "Generated" in r.text
    assert "UTC" in r.text


def test_number_banner_is_human_and_machine_readable(client):
    tc, _store = client
    r = _post(tc, target_json=_target_json())
    assert r.status_code == 200
    assert "Prediction issued" in r.text
    assert "banner-number" in r.text
    assert "NUMBER" in r.text


def test_number_with_warning_banner_lists_the_warning(client):
    tc, _store = client
    r = _post(
        tc,
        target_json=_target_json(process_params={"ph_setpoint": 9.0, "temperature_C": 37.0}),
    )
    assert r.status_code == 200, r.text
    assert "Prediction issued with warnings" in r.text
    assert "banner-warning" in r.text
    assert "NUMBER_WITH_WARNING" in r.text
    assert "ph_setpoint" in r.text  # the out-of-range param is named


def test_refusal_banner_names_the_reason(client):
    tc, _store = client
    # ratio ~17x: past REFUSE_RATIO_MULT (5x) of the ~3.33x reference, but
    # still under the gate's 20x hard cap - a decide()-level refusal, not a
    # gate failure, so this is 200 HTML, not 422.
    r = _post(tc, target_json=_target_json(scale_L=5000.0 * 17.0))
    assert r.status_code == 200, r.text
    assert "No prediction:" in r.text
    assert "banner-refusal" in r.text
    assert "REFUSAL" in r.text


def test_plan_json_is_isolated_in_a_page_break_appendix():
    """Everything except the raw normalize-plan JSON belongs on printed page
    1; the JSON lives in an appendix that the print CSS forces onto its own
    page (`.appendix { break-before: page }`). A real print render is a
    manual QA step (see docs/SCALE_READOUT.md); this pins the structure that
    makes it work, and runs everywhere."""
    from kalos.portal.scale_routes import render_readout_html
    from kalos.scale.readout import TargetSpec, build_readout

    df = pd.read_csv(_DEMO_CSV)
    target = TargetSpec(
        scale_L=7500.0,
        agitation_rpm=150.0,
        airflow_L_per_min=300.0,
        process_params={"ph_setpoint": 7.2, "temperature_C": 37.0},
    )
    readout = build_readout(df, "titer_g_per_L", ["ph_setpoint", "temperature_C"], target)
    page = render_readout_html(readout)

    assert re.search(r"\.appendix\s*\{[^}]*break-before:\s*page", page)
    appendix_at = page.index('<section class="appendix">')
    page_one, appendix = page[:appendix_at], page[appendix_at:]
    assert "&quot;columns&quot;" not in page_one and '"columns":' not in page_one
    assert "&quot;columns&quot;" in appendix or '"columns":' in appendix
    assert "Development decision support only" in page_one
    assert "Baseline comparison".upper() in page_one.upper()
    assert "Physics assumptions".upper() in page_one.upper()


def test_large_scales_never_render_in_scientific_notation():
    """A 40,000 L target must read "40,000 L", not "4e+04 L"."""
    from kalos.portal.scale_routes import _fmt_sig4

    assert _fmt_sig4(40000.0) == "40,000"
    assert _fmt_sig4(125000.0) == "125,000"
    assert _fmt_sig4(7500.0) == "7500"
    assert _fmt_sig4(1.0 / 3.0) == "0.3333"
    assert "e+" not in _fmt_sig4(1e6)


def test_sparse_tech_transfer_sheet_returns_a_200_refusal_page(client):
    """The shape real tech-transfer data has (many bench runs, one or two
    runs per large scale) must get a readable refusal page with its
    evidence and a data plan - not a 422 and not a number."""
    from test_scale_readout import _sparse_sheet  # pytest puts tests/ on sys.path

    tc, _ = client
    csv_bytes = _sparse_sheet().to_csv(index=False).encode("utf-8")
    r = _post(tc, target_json=_target_json(scale_L=42000.0, airflow_L_per_min=1500.0, process_params={"ph_setpoint": 7.0, "temperature_C": 36.7}), csv_bytes=csv_bytes)
    assert r.status_code == 200, r.text
    page = r.text
    assert "No prediction:" in page
    # every refusal reason is listed, not just the first
    assert "pooled ladder residual(s); need at least 10" in page
    assert "to license a reference step ratio" in page
    assert "What it would take" in page
    assert "more run(s) at any scale of 15 L or larger" in page
    assert "too few to judge" in page
    # the pooled verdict is withheld on thin evidence
    assert "beats both baselines</span><span class=\"v\">yes" not in page


def test_scale_only_page_lists_unused_constants_without_values():
    """Scale-only mode must not print values for physics constants the model
    never saw; it names them once on a single line instead."""
    from kalos.portal.scale_routes import render_readout_html
    from kalos.scale.readout import TargetSpec, build_readout

    df = pd.read_csv(_DEMO_CSV).drop(columns=["agitation_rpm", "airflow_L_per_min"])
    target = TargetSpec(scale_L=7500.0, process_params={"ph_setpoint": 7.2, "temperature_C": 37.0})
    page = render_readout_html(build_readout(df, "titer_g_per_L", ["ph_setpoint", "temperature_C"], target))
    assert "Not used by the scale-only model:" in page
    assert "power_number" in page
    assert "(not used" not in page  # no per-constant "not used" rows with values

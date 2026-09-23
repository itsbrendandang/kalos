"""Kalos portal - `POST /api/scale/readout`: the Scale-Up Readout (D5/D6 in
the design's decision ledger). A stateless multipart endpoint mirroring the
router pattern in `kalos.portal.campaign_routes` / `kalos.portal.experiments`:
upload a multi-scale run sheet plus a target JSON, get back a single
self-contained HTML page (print CSS, one page) with the prediction,
interval, baseline comparison, and provenance - or a 422 naming the failed
gate check.

STATELESS BY DESIGN (D6): nothing from this route is written to disk or a
database. The run sheet and target are consumed for exactly one response and
then discarded; a prospect who wants the page again has to re-upload (see
`kalos.scale.readout`'s module docstring, and the reproducibility scope in
the design doc).
"""
from __future__ import annotations

import functools
import html as html_lib
import json
import logging
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from kalos.portal.auth import WRITE, Principal, require_scope
from kalos.portal.busy import RETRY_AFTER_SECONDS, AnalysisBusy, run_exclusively
from kalos.portal.uploads import _ERR_PARSE, UploadRejected, _parse_upload
from kalos.scale.readout import NUMBER, NUMBER_WITH_WARNING, REFUSAL, TargetSpec, build_readout
from kalos.scale.readout import format_liters_plain as _format_liters_plain

log = logging.getLogger("kalos.portal")

router = APIRouter()

_TEMPLATE = Template(
    (Path(__file__).parent / "templates" / "readout.html").read_text(encoding="utf-8")
)

# Fixed intended-use statement (design doc step 8) - printed verbatim, never
# built from a format string that could drift.
INTENDED_USE_STATEMENT = (
    "Development decision support only. Not a GMP or regulatory record. "
    "Not validated under 21 CFR Part 11."
)

_ERR_INVALID_TARGET = "invalid target JSON"


def _esc(value: Any) -> str:
    """HTML-escape any value that might carry user-supplied text (a column
    name, a check reason, a physics-override value) before it reaches the
    page."""
    return html_lib.escape(str(value))


class _TargetJsonError(ValueError):
    """The `target` form field was not valid JSON, or was missing a
    required key - a 422, not a gate failure."""


def _parse_target(raw: str) -> tuple[TargetSpec, str, list[str], dict[str, float] | None]:
    """Parse the `target` form field into `(TargetSpec, target_column,
    process_columns, physics_overrides)`.

    Expected shape:
        {
          "scale_L": 7500.0,
          "agitation_rpm": 150.0,
          "airflow_L_per_min": 300.0,
          "target_column": "titer_g_per_L",
          "process_params": {"ph_setpoint": 7.0, "temperature_C": 37.0},
          "physics_overrides": {"power_number": 6.0}   // optional
        }

    `process_columns` is taken from `process_params`' keys - the same set
    `kalos.scale.readout.gate` requires to match exactly, so there is only
    one place a caller names the process parameters.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _TargetJsonError(f"{_ERR_INVALID_TARGET}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise _TargetJsonError(f"{_ERR_INVALID_TARGET}: expected a JSON object")

    required = ("scale_L", "agitation_rpm", "airflow_L_per_min", "target_column", "process_params")
    missing = [k for k in required if k not in data]
    if missing:
        raise _TargetJsonError(f"{_ERR_INVALID_TARGET}: missing key(s) {missing}")

    process_params = data["process_params"]
    if not isinstance(process_params, dict) or not process_params:
        raise _TargetJsonError(f"{_ERR_INVALID_TARGET}: 'process_params' must be a non-empty object")

    physics_overrides = data.get("physics_overrides")
    if physics_overrides is not None and not isinstance(physics_overrides, dict):
        raise _TargetJsonError(f"{_ERR_INVALID_TARGET}: 'physics_overrides' must be an object")

    try:
        target = TargetSpec(
            scale_L=float(data["scale_L"]),
            agitation_rpm=float(data["agitation_rpm"]),
            airflow_L_per_min=float(data["airflow_L_per_min"]),
            process_params={str(k): float(v) for k, v in process_params.items()},
        )
    except (TypeError, ValueError) as exc:
        raise _TargetJsonError(f"{_ERR_INVALID_TARGET}: {exc}") from exc

    target_column = str(data["target_column"])
    process_columns = list(target.process_params)
    return target, target_column, process_columns, physics_overrides


# --- HTML rendering ----------------------------------------------------------- #

# Rendering-only formatting rules (not part of the data contract in
# `kalos.scale.readout` - the result dict keeps full-precision floats;
# these format specs are applied only when building the page):
#   - MAE / baseline / prediction / interval bound: fixed 3 decimals.
#   - step ratio: fixed 2 decimals, "x" suffix.
#   - physics assumption: up to 4 significant digits.
_FMT_METRIC = "{:.3f}"
_FMT_RATIO = "{:.2f}x"


def _fmt_sig4(value: float) -> str:
    """Up to 4 significant digits, never scientific notation: plant scales
    reach 10^4-10^5 L, and "4e+04 L" is not something to put in front of a
    process lead. Magnitudes of 10^4 and above print as grouped integers
    ("40,000"); smaller values keep `{:.4g}` ("0.3333", "7.2", "1000").
    Delegates to `kalos.scale.readout.format_liters_plain` so a data-plan
    liters figure (computed in that module) and every other rendered scale
    figure on this page always agree."""
    return _format_liters_plain(value)


def _kv(label: str, value: Any, *, mono: bool = False) -> str:
    cls = "kv kv-hash" if mono else "kv"
    return f'<div class="{cls}"><span class="k">{_esc(label)}</span><span class="v">{_esc(value)}</span></div>'


def _target_inputs_section(readout: dict[str, Any]) -> str:
    inputs = readout["target_inputs"]
    rows = [
        ("Target scale", f"{_fmt_sig4(inputs['scale_L'])} L"),
        ("Agitation", f"{_fmt_sig4(inputs['agitation_rpm'])} rpm"),
        ("Airflow", f"{_fmt_sig4(inputs['airflow_L_per_min'])} L/min"),
    ]
    for name, value in inputs["process_params"].items():
        rows.append((name, _fmt_sig4(value)))
    kvs = "".join(_kv(k, v) for k, v in rows)
    return f'<h2>Target inputs</h2><div class="grid">{kvs}</div>'


def _prediction_labels(readout: dict[str, Any]) -> tuple[str, str]:
    """`(prediction_label, interval_label)` - named after the target column,
    tagged with its resolved unit when the normalize plan found one, and the
    target scale, per the coordinator's "name the prediction" request."""
    target_column = readout.get("target_column") or "target"
    unit = (readout.get("provenance") or {}).get("target_column_unit")
    unit_part = f" ({unit})" if unit else ""
    scale = readout["target_inputs"]["scale_L"]
    subject = f"{target_column}{unit_part} at {_fmt_sig4(scale)} L"
    return f"Predicted {subject}", f"Interval ({readout['interval_label']}) for {subject}"


def _prediction_section(readout: dict[str, Any]) -> str:
    if readout["prediction"] is None:
        return ""
    lo, hi = readout["interval"]
    prediction_label, interval_label = _prediction_labels(readout)
    return (
        "<h2>Prediction</h2>"
        + _kv(prediction_label, _FMT_METRIC.format(readout["prediction"]))
        + _kv(interval_label, f"[{_FMT_METRIC.format(lo)}, {_FMT_METRIC.format(hi)}]")
    )


_BANNER_CSS = {NUMBER: "banner-number", NUMBER_WITH_WARNING: "banner-warning", REFUSAL: "banner-refusal"}


def _decision_banner(readout: dict[str, Any]) -> str:
    decision = readout["decision"]
    reasons = readout.get("reasons") or []
    extra = ""
    if decision == NUMBER_WITH_WARNING:
        headline = "Prediction issued with warnings"
        if reasons:
            items = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
            extra = f'<ul class="reasons">{items}</ul>'
    elif decision == REFUSAL:
        joined = "; ".join(reasons) if reasons else "no reason given"
        headline = f"No prediction: {_esc(joined)}"
    else:
        headline = "Prediction issued"
    css = _BANNER_CSS.get(decision, "banner-refusal")
    return (
        f'<div class="banner {css}">'
        f'<span class="banner-headline">{headline}</span>'
        f'<span class="banner-machine">{_esc(decision)}</span>'
        "</div>"
        f"{extra}"
    )


def _rungs_table(readout: dict[str, Any]) -> str:
    rungs = readout.get("rungs") or []
    skipped = readout.get("skipped_rungs") or []
    parts = ["<h2>Backtest ladder</h2>"]
    if rungs:
        rows = "".join(
            "<tr>"
            f'<td>{_fmt_sig4(r["scale_L"])} L</td><td>{_FMT_RATIO.format(r["step_ratio"])}</td><td>{r["n"]}</td>'
            f'<td>{_FMT_METRIC.format(r["mae"])}</td><td>{_FMT_METRIC.format(r["naive_mean_mae"])}</td>'
            f'<td>{_FMT_METRIC.format(r["naive_nn_mae"])}</td>'
            f'<td>{"too few to judge" if r["too_few_to_judge"] else ("yes" if r["beats_both"] else "no")}</td>'
            "</tr>"
            for r in rungs
        )
        parts.append(
            "<table><thead><tr><th>Scale</th><th>Step ratio</th><th>n</th><th>MAE</th>"
            "<th>naive_mean MAE</th><th>naive_nn MAE</th><th>beats both</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        parts.append('<p class="muted">No rungs were evaluated.</p>')
    if skipped:
        items = "".join(
            f'<li>scale {_fmt_sig4(s["scale_L"])} L skipped: {_esc(s["reason"])}</li>' for s in skipped
        )
        parts.append(f'<p class="muted">Skipped rungs:</p><ul class="skipped-reasons">{items}</ul>')
    return "".join(parts)


def _ratio_section(readout: dict[str, Any]) -> str:
    reference = readout.get("reference_ratio")
    requested = readout.get("requested_ratio")
    reference_str = _FMT_RATIO.format(reference) if reference is not None else "n/a (no rung beat both baselines)"
    requested_str = _FMT_RATIO.format(requested) if requested is not None else "n/a"
    out_of_range = readout.get("out_of_range_params") or []
    extra = ""
    if out_of_range:
        extra = f'<p class="muted">Out-of-range process parameter(s): {_esc(", ".join(out_of_range))}</p>'
    return (
        "<h2>Step ratio</h2>"
        + _kv("Reference (backtested) ratio", reference_str)
        + _kv("Requested (target) ratio", requested_str)
        + extra
    )


def _baseline_section(readout: dict[str, Any]) -> str:
    comparison = readout.get("baseline_comparison")
    if comparison is None or comparison["n_residuals"] == 0:
        return ""
    return (
        "<h2>Baseline comparison (pooled ladder residuals)</h2>"
        + _kv("Model MAE", _FMT_METRIC.format(comparison["pooled_mae"]))
        + _kv("naive_mean MAE", _FMT_METRIC.format(comparison["pooled_naive_mean_mae"]))
        + _kv("naive_nn MAE", _FMT_METRIC.format(comparison["pooled_naive_nn_mae"]))
        + _kv("beats both baselines", "yes" if comparison["pooled_beats_both"] else "no")
        + _kv("pooled residuals (n)", comparison["n_residuals"])
    )


def _data_plan_section(readout: dict[str, Any]) -> str:
    """"What it would take": plain, specific lines computed in
    `kalos.scale.readout._build_data_plan` (residual shortfall, no
    licensing rung, ratio too far from the reference) - rendered whenever
    `build_readout` populated `data_plan` (REFUSAL and NUMBER_WITH_WARNING;
    empty otherwise)."""
    plan = readout.get("data_plan") or []
    if not plan:
        return ""
    items = "".join(f"<li>{_esc(p)}</li>" for p in plan)
    return f'<h2>What it would take</h2><ul class="data-plan">{items}</ul>'


def _physics_section(readout: dict[str, Any]) -> str:
    assumptions = readout.get("physics_assumptions") or {}
    rows = "".join(
        _kv(f"{name} ({info['source']})", _fmt_sig4(info["value"]))
        for name, info in sorted(assumptions.items())
    )
    return f'<h2>Physics assumptions</h2><div class="grid">{rows}</div>'


def _format_const_value(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    return _fmt_sig4(value)


def _constants_table(constants: dict[str, Any], *, per_row: int = 4) -> str:
    """The decision-table constants as one compact multi-column table
    (name/value pairs packed `per_row` to a row), not one dt/dd pair per
    line - this is what keeps page 1 to a single page."""
    items = sorted(constants.items())
    rows = []
    for i in range(0, len(items), per_row):
        cells = "".join(
            f'<td class="name">{_esc(name)}</td><td class="value">{_esc(_format_const_value(value))}</td>'
            for name, value in items[i : i + per_row]
        )
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="const-table"><tbody>{"".join(rows)}</tbody></table>'


def _provenance_section(readout: dict[str, Any]) -> str:
    """Page-1 provenance: full hashes on their own compact monospace lines,
    a one-line pin for the normalize plan (column count + its own hash -
    the full plan JSON lives in the page-2 appendix), and the constants as
    one compact table."""
    p = readout.get("provenance") or {}
    constants = p.get("constants") or {}
    rows = (
        _kv("engine version", p.get("engine_version"))
        + _kv("kalos git SHA", p.get("kalos_git_sha"), mono=True)
        + _kv("candidate", p.get("candidate"))
        + _kv("alpha", p.get("alpha"))
        + _kv("seed", p.get("seed"))
        + _kv("resolved scale_L unit", p.get("resolved_scale_unit"))
    )
    plan_pin = f"{p.get('normalize_plan_n_columns')} columns, SHA-256 {p.get('normalize_plan_sha256')}"
    hash_rows = (
        _kv("raw upload SHA-256", p.get("raw_upload_sha256"), mono=True)
        + _kv("normalized frame SHA-256", p.get("normalized_frame_sha256"), mono=True)
        + _kv("normalize plan", plan_pin, mono=True)
    )
    return (
        "<h2>Provenance</h2>"
        f'<div class="grid">{rows}</div>'
        f"{hash_rows}"
        f"{_constants_table(constants)}"
    )


def _provenance_appendix(readout: dict[str, Any]) -> str:
    """Page 2: the full normalize plan JSON - the only thing NOT required
    to fit on page 1 (page 1 already pins its hash and column count in
    `_provenance_section`)."""
    plan_json = (readout.get("provenance") or {}).get("normalize_plan_json") or ""
    return (
        '<section class="appendix">'
        "<h2>Provenance appendix: normalize plan (full JSON)</h2>"
        f'<div class="plan-json">{_esc(plan_json)}</div>'
        "</section>"
    )


def render_readout_html(readout: dict[str, Any]) -> str:
    """Render `build_readout`'s output dict into the single self-contained
    HTML page (see `kalos/portal/templates/readout.html` for the skeleton
    and print CSS). Every user-supplied string is HTML-escaped via `_esc`
    before it reaches the page.

    Page 1 holds everything except the raw normalize-plan JSON, which is
    the one section moved to a page-2 "Provenance appendix" (`break-before:
    page` in the template's print CSS) - page 1 still pins that JSON's
    identity via its SHA-256 and column count in `_provenance_section`.
    """
    page1 = "".join(
        s
        for s in (
            _decision_banner(readout),
            _prediction_section(readout),
            _target_inputs_section(readout),
            _rungs_table(readout),
            _ratio_section(readout),
            _baseline_section(readout),
            _data_plan_section(readout),
            _physics_section(readout),
            _provenance_section(readout),
        )
        if s
    )
    intended_use = f'<p class="intended-use">{_esc(INTENDED_USE_STATEMENT)}</p>'
    body = page1 + intended_use + _provenance_appendix(readout)
    # The dict keeps ISO 8601 ("...T18:32:17+00:00"); the page shows
    # "2026-09-23 18:32:17" next to the template's own "UTC" label.
    generated_at = _esc(str(readout.get("generated_at_utc", "")).replace("T", " ").removesuffix("+00:00"))
    return _TEMPLATE.substitute(body=body, generated_at=generated_at)


# --- route -------------------------------------------------------------------- #


@router.post("/api/scale/readout")
async def scale_readout(
    file: UploadFile = File(...),
    target: str = Form(...),
    principal: Principal = Depends(require_scope(WRITE)),
) -> Response:
    """Upload a multi-scale run sheet plus a target JSON, get back the
    Scale-Up Readout as a single HTML page.

    Stateless (D6): nothing is stored server-side, not even on success.
    """
    try:
        target_spec, target_column, process_columns, physics_overrides = _parse_target(target)
    except _TargetJsonError as exc:
        return JSONResponse({"failed_check": "invalid_target", "detail": str(exc)}, status_code=422)

    raw = await file.read()
    try:
        df = await run_in_threadpool(_parse_upload, raw)
    except UploadRejected as rej:
        log.warning("scale readout upload rejected: %s", rej)
        return JSONResponse({"error": str(rej)}, status_code=400)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, UnicodeError):
        log.exception("failed to parse an uploaded run sheet for the scale readout")
        return JSONResponse({"error": _ERR_PARSE}, status_code=400)
    except Exception:  # noqa: BLE001 - normalized to a generic 400, same as /api/run
        log.exception("failed to parse an uploaded run sheet for the scale readout")
        return JSONResponse({"error": _ERR_PARSE}, status_code=400)

    try:
        readout = await run_in_threadpool(
            run_exclusively(
                functools.partial(
                    build_readout,
                    df,
                    target_column,
                    process_columns,
                    target_spec,
                    physics_overrides=physics_overrides,
                    raw_bytes=raw,
                )
            )
        )
    except AnalysisBusy as busy:
        return JSONResponse(
            {"error": str(busy)}, status_code=503, headers={"Retry-After": str(RETRY_AFTER_SECONDS)}
        )
    except ValueError as exc:
        # e.g. a malformed process_params/target mismatch surfaced by gate().
        return JSONResponse({"failed_check": "invalid_target", "detail": str(exc)}, status_code=422)

    if not readout["gate"]["passed"]:
        return JSONResponse(
            {"failed_check": readout["gate"]["failed_check"], "detail": readout["gate"]["detail"]},
            status_code=422,
        )

    return HTMLResponse(render_readout_html(readout), status_code=200)


__all__ = ["router", "render_readout_html", "INTENDED_USE_STATEMENT"]

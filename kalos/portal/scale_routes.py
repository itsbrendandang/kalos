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
from kalos.scale.readout import NUMBER, NUMBER_WITH_WARNING, TargetSpec, build_readout

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


def _target_inputs_section(readout: dict[str, Any]) -> str:
    inputs = readout["target_inputs"]
    rows = [
        ("Target scale", f"{inputs['scale_L']:g} L"),
        ("Agitation", f"{inputs['agitation_rpm']:g} rpm"),
        ("Airflow", f"{inputs['airflow_L_per_min']:g} L/min"),
    ]
    for name, value in inputs["process_params"].items():
        rows.append((_esc(name), f"{value:g}"))
    kvs = "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in rows)
    return f'<h2>Target inputs</h2><div class="grid">{kvs}</div>'


def _prediction_section(readout: dict[str, Any]) -> str:
    if readout["prediction"] is None:
        return ""
    lo, hi = readout["interval"]
    return (
        "<h2>Prediction</h2>"
        f'<div class="kv"><span class="k">Predicted value</span><span class="v">{readout["prediction"]:.4g}</span></div>'
        f'<div class="kv"><span class="k">Interval ({_esc(readout["interval_label"])})</span>'
        f'<span class="v">[{lo:.4g}, {hi:.4g}]</span></div>'
    )


def _decision_banner(readout: dict[str, Any]) -> str:
    decision = readout["decision"]
    css = {"NUMBER": "banner-number", "NUMBER_WITH_WARNING": "banner-warning", "REFUSAL": "banner-refusal"}
    label = {
        NUMBER: "NUMBER",
        NUMBER_WITH_WARNING: "NUMBER, WITH WARNING",
        "REFUSAL": "REFUSAL",
    }.get(decision, _esc(decision))
    banner = f'<div class="banner {css.get(decision, "banner-refusal")}">{label}</div>'
    reasons = readout.get("reasons") or []
    if reasons:
        items = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
        banner += f'<ul class="reasons">{items}</ul>'
    return banner


def _rungs_table(readout: dict[str, Any]) -> str:
    rungs = readout.get("rungs") or []
    skipped = readout.get("skipped_rungs") or []
    parts = ["<h2>Backtest ladder</h2>"]
    if rungs:
        rows = "".join(
            "<tr>"
            f'<td>{r["scale_L"]:g} L</td><td>{r["step_ratio"]:.2f}x</td><td>{r["n"]}</td>'
            f'<td>{r["mae"]:.4g}</td><td>{r["naive_mean_mae"]:.4g}</td><td>{r["naive_nn_mae"]:.4g}</td>'
            f'<td>{"yes" if r["beats_both"] else "no"}</td>'
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
        items = "".join(f'<li>scale {s["scale_L"]:g} L skipped: {_esc(s["reason"])}</li>' for s in skipped)
        parts.append(f'<p class="muted">Skipped rungs:</p><ul class="reasons">{items}</ul>')
    return "".join(parts)


def _ratio_section(readout: dict[str, Any]) -> str:
    reference = readout.get("reference_ratio")
    requested = readout.get("requested_ratio")
    reference_str = f"{reference:.2f}x" if reference is not None else "n/a (no rung beat both baselines)"
    requested_str = f"{requested:.2f}x" if requested is not None else "n/a"
    out_of_range = readout.get("out_of_range_params") or []
    extra = ""
    if out_of_range:
        extra = f'<p class="muted">Out-of-range process parameter(s): {_esc(", ".join(out_of_range))}</p>'
    return (
        "<h2>Step ratio</h2>"
        f'<div class="kv"><span class="k">Reference (backtested) ratio</span><span class="v">{reference_str}</span></div>'
        f'<div class="kv"><span class="k">Requested (target) ratio</span><span class="v">{requested_str}</span></div>'
        f"{extra}"
    )


def _baseline_section(readout: dict[str, Any]) -> str:
    comparison = readout.get("baseline_comparison")
    if comparison is None:
        return ""
    return (
        "<h2>Baseline comparison (pooled ladder residuals)</h2>"
        f'<div class="kv"><span class="k">Model MAE</span><span class="v">{comparison["pooled_mae"]:.4g}</span></div>'
        f'<div class="kv"><span class="k">naive_mean MAE</span><span class="v">{comparison["pooled_naive_mean_mae"]:.4g}</span></div>'
        f'<div class="kv"><span class="k">naive_nn MAE</span><span class="v">{comparison["pooled_naive_nn_mae"]:.4g}</span></div>'
        f'<div class="kv"><span class="k">beats both baselines</span><span class="v">{"yes" if comparison["pooled_beats_both"] else "no"}</span></div>'
        f'<div class="kv"><span class="k">pooled residuals (n)</span><span class="v">{comparison["n_residuals"]}</span></div>'
    )


def _physics_section(readout: dict[str, Any]) -> str:
    assumptions = readout.get("physics_assumptions") or {}
    rows = "".join(
        f'<div class="kv"><span class="k">{_esc(name)} ({_esc(info["source"])})</span>'
        f'<span class="v">{_esc(info["value"])}</span></div>'
        for name, info in sorted(assumptions.items())
    )
    return f'<h2>Physics assumptions</h2><div class="grid">{rows}</div>'


def _provenance_section(readout: dict[str, Any]) -> str:
    p = readout.get("provenance") or {}
    constants = p.get("constants") or {}
    const_rows = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in sorted(constants.items()))
    return (
        "<h2>Provenance</h2>"
        '<dl class="provenance">'
        f"<dt>kalos git SHA</dt><dd class=\"mono\">{_esc(p.get('kalos_git_sha'))}</dd>"
        f"<dt>candidate</dt><dd>{_esc(p.get('candidate'))}</dd>"
        f"<dt>raw upload SHA-256</dt><dd class=\"mono\">{_esc(p.get('raw_upload_sha256'))}</dd>"
        f"<dt>normalized frame SHA-256</dt><dd class=\"mono\">{_esc(p.get('normalized_frame_sha256'))}</dd>"
        f"<dt>alpha</dt><dd>{_esc(p.get('alpha'))}</dd>"
        f"<dt>seed</dt><dd>{_esc(p.get('seed'))}</dd>"
        f"<dt>resolved scale_L unit</dt><dd>{_esc(p.get('resolved_scale_unit'))}</dd>"
        f"<dt>normalize plan (JSON)</dt><dd class=\"mono\">{_esc(p.get('normalize_plan_json'))}</dd>"
        f"{const_rows}"
        "</dl>"
    )


def render_readout_html(readout: dict[str, Any]) -> str:
    """Render `build_readout`'s output dict into the single self-contained
    HTML page (see `kalos/portal/templates/readout.html` for the skeleton
    and print CSS). Every user-supplied string is HTML-escaped via `_esc`
    before it reaches the page.
    """
    sections = [
        _decision_banner(readout),
        _prediction_section(readout),
        _target_inputs_section(readout),
        _rungs_table(readout),
        _ratio_section(readout),
        _baseline_section(readout),
        _physics_section(readout),
        _provenance_section(readout),
    ]
    body = "".join(s for s in sections if s)
    return _TEMPLATE.substitute(body=body, intended_use=_esc(INTENDED_USE_STATEMENT))


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

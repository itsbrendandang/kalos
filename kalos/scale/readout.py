"""The Scale-Up Readout: gate -> ladder backtest -> decision -> interval.

This module answers one question honestly: "what will this KPI read at a
target scale we have not run yet, and how much should you trust that
number." It is a thin orchestration layer over code that already exists
(`kalos.scale.evaluation.leave_one_scale_out_report`,
`kalos.scale.transfer.ScaleUpTransferModel`, `kalos.core.conformal`) - see
the design doc referenced in this repo's PR history (Kalos SaaS: Scale-Up
Readout as the first wedge) for the full decision ledger (R1-R8/D4-D11) this
module's behavior implements.

PIPELINE (see `build_readout`):
  1. `gate` - six named checks, in order; the first one that fails stops the
     pipeline and names itself, so a caller never sees a silent partial
     answer.
  2. `ladder` - a nested leave-last-scale-out backtest: for every scale
     (except the smallest two, which cannot form a >=2-scale training set),
     fit on every smaller scale and predict that scale, keeping the
     per-row residuals and the naive-baseline comparison. This measures the
     model's demonstrated scale-transfer skill, at whatever step ratios the
     uploaded sheet actually lets it test.
  3. `decide` - NUMBER / NUMBER_WITH_WARNING / REFUSAL from the ladder's
     evidence. A number is NEVER issued without an interval.
  4. Final fit (`ScaleUpTransferModel`, candidate `v0`, EXPLICIT bounds
     spanning training AND target) -> `predict` at the target -> a
     split-conformal interval calibrated on the ladder's pooled residuals
     (labeled "approximate coverage": exchangeability does not hold across
     an untested scale, so this is not called calibrated).

UNITS. `scale_L` is liters, `agitation_rpm` is rpm, `airflow_L_per_min` is
liters per minute - the same units `kalos.scale.features` assumes.

Every threshold below is a named module-level constant so the readout can
print exactly what gated it, rather than a caller having to reverse-engineer
a number baked into a comparison.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kalos.core.conformal import conformal_interval, q_from_residuals
from kalos.normalize import NormalizationPlan, apply_plan, offline_plan
from kalos.normalize.units import base_unit_label, canonical_suffix

from .evaluation import leave_one_scale_out_report
from .features import GeometryAssumptions, PowerNumberAssumption, VantRietParams
from .transfer import DEFAULT_SCALE_FEATURE_CONFIG, ScaleFeatureConfig, ScaleUpTransferModel, build_feature_matrix

# --- named constants (every one of these is printed in the readout's ----- #
# --- provenance section - see Code Quality review finding 1). ----------- #
MIN_SCALES = 3
MIN_RUNS_PER_SCALE = 3
MIN_RUNG_TARGET_ROWS = 10
MIN_RUNG_TRAIN_SCALES = 2
MIN_RUNG_TRAIN_ROWS = 4
MAX_DROPPED_FRACTION = 0.20
HARD_CAP_RATIO = 20.0
WARN_RATIO_MULT = 2.0
REFUSE_RATIO_MULT = 5.0
ALPHA = 0.1
MIN_RESIDUALS = 10
CANDIDATE = "v0"

_ERR_INTERPOLATION = "interpolation is out of scope for this readout"

CONSTANTS: dict[str, float | str] = {
    "MIN_SCALES": MIN_SCALES,
    "MIN_RUNS_PER_SCALE": MIN_RUNS_PER_SCALE,
    "MIN_RUNG_TARGET_ROWS": MIN_RUNG_TARGET_ROWS,
    "MIN_RUNG_TRAIN_SCALES": MIN_RUNG_TRAIN_SCALES,
    "MIN_RUNG_TRAIN_ROWS": MIN_RUNG_TRAIN_ROWS,
    "MAX_DROPPED_FRACTION": MAX_DROPPED_FRACTION,
    "HARD_CAP_RATIO": HARD_CAP_RATIO,
    "WARN_RATIO_MULT": WARN_RATIO_MULT,
    "REFUSE_RATIO_MULT": REFUSE_RATIO_MULT,
    "ALPHA": ALPHA,
    "MIN_RESIDUALS": MIN_RESIDUALS,
    "CANDIDATE": CANDIDATE,
}


# --- input contract -------------------------------------------------------- #


@dataclass(frozen=True)
class TargetSpec:
    """The target scale plus its planned operating conditions.

    `scale_L`: target vessel volume, liters.
    `agitation_rpm`: planned impeller speed at the target scale, rpm.
    `airflow_L_per_min`: planned sparge/gas flow at the target scale, liters
    per minute.
    `process_params`: the recipe inputs to be run at the target scale, one
    entry per column in `process_columns` (e.g. `{"ph_setpoint": 7.0,
    "temperature_C": 37.0}`) - checked against the trained range separately
    from the scale axis (see `decide`).
    """

    scale_L: float
    agitation_rpm: float
    airflow_L_per_min: float
    process_params: dict[str, float]

    def to_frame(self, config: ScaleFeatureConfig) -> pd.DataFrame:
        """A single-row frame in the shape `build_feature_matrix` expects."""
        row = dict(self.process_params)
        row[config.scale_column] = self.scale_L
        row[config.agitation_column] = self.agitation_rpm
        row[config.airflow_column] = self.airflow_L_per_min
        return pd.DataFrame([row])


# --- gate ------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateResult:
    """The outcome of `gate`. `clean_df` (only set when `passed`) is `df`
    with non-finite rows in the required columns already dropped - the
    same frame `ladder`/the final fit should be run on, so the row counts
    the gate promised are the row counts the pipeline actually sees."""

    passed: bool
    failed_check: str | None
    detail: str
    n_rows_dropped: int
    dropped_fraction: float
    target_ratio: float | None
    clean_df: pd.DataFrame | None = None


def _required_columns(target_column: str, process_columns: Sequence[str], config: ScaleFeatureConfig) -> list[str]:
    return [target_column, config.scale_column, config.agitation_column, config.airflow_column, *process_columns]


def gate(
    df: pd.DataFrame,
    target_column: str,
    process_columns: Sequence[str],
    target: TargetSpec,
    *,
    config: ScaleFeatureConfig = DEFAULT_SCALE_FEATURE_CONFIG,
) -> GateResult:
    """Six named checks, in order; the first failure stops the pipeline.

    1. `required_columns` - every required column is present.
    2. `min_scales_and_runs` - at least `MIN_SCALES` distinct scales, each
       with at least `MIN_RUNS_PER_SCALE` runs.
    3. `min_rung_target_rows` - at least `MIN_RUNG_TARGET_ROWS` rows at the
       third-smallest scale and above (the rungs `ladder` can actually
       backtest), so passing this guarantees `ladder` can reach
       `MIN_RESIDUALS` pooled residuals.
    4. `non_finite_dropped_fraction` - the fraction of rows dropped for a
       non-finite required column is at most `MAX_DROPPED_FRACTION`.
    5. `target_not_interpolation` - the target scale is strictly larger than
       every trained scale.
    6. `target_ratio_hard_cap` - the target/largest-trained-scale ratio is
       at most `HARD_CAP_RATIO`.

    If `target.process_params`' keys do not exactly match `process_columns`,
    raises `ValueError` (a caller/target-JSON error, not a gate failure - the
    portal route maps this to its own 422 before ever calling `gate`).
    """
    if set(target.process_params) != set(process_columns):
        raise ValueError(
            f"target.process_params keys {sorted(target.process_params)} do not match "
            f"process_columns {sorted(process_columns)}"
        )

    required = _required_columns(target_column, process_columns, config)
    missing = [c for c in required if c not in df.columns]
    if missing:
        return GateResult(
            passed=False,
            failed_check="required_columns",
            detail=f"missing required column(s): {missing}",
            n_rows_dropped=0,
            dropped_fraction=0.0,
            target_ratio=None,
        )

    n_total = len(df)
    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    dropped_fraction = (n_dropped / n_total) if n_total else 1.0
    clean_df = df.loc[finite_mask].reset_index(drop=True)

    scale = clean_df[config.scale_column].to_numpy(dtype=float)
    scale = np.round(scale, 6)
    unique_scales = np.unique(scale)

    if len(unique_scales) < MIN_SCALES:
        return GateResult(
            passed=False,
            failed_check="min_scales_and_runs",
            detail=f"only {len(unique_scales)} distinct scale(s) present; need at least {MIN_SCALES}",
            n_rows_dropped=n_dropped,
            dropped_fraction=dropped_fraction,
            target_ratio=None,
        )
    counts = {float(s): int((scale == s).sum()) for s in unique_scales}
    under = {s: n for s, n in counts.items() if n < MIN_RUNS_PER_SCALE}
    if under:
        return GateResult(
            passed=False,
            failed_check="min_scales_and_runs",
            detail=f"scale(s) below {MIN_RUNS_PER_SCALE} runs: {under}",
            n_rows_dropped=n_dropped,
            dropped_fraction=dropped_fraction,
            target_ratio=None,
        )

    # rung targets are the third-smallest scale and up (see the gate's
    # own docstring and R4/D7 in the design's decision ledger).
    rung_target_scales = sorted(unique_scales)[2:]
    n_rung_rows = int(sum(counts[float(s)] for s in rung_target_scales))
    if n_rung_rows < MIN_RUNG_TARGET_ROWS:
        return GateResult(
            passed=False,
            failed_check="min_rung_target_rows",
            detail=(
                f"only {n_rung_rows} row(s) at the third-smallest scale and above; "
                f"need at least {MIN_RUNG_TARGET_ROWS} so the ladder can reach {MIN_RESIDUALS} residuals"
            ),
            n_rows_dropped=n_dropped,
            dropped_fraction=dropped_fraction,
            target_ratio=None,
        )

    if dropped_fraction > MAX_DROPPED_FRACTION:
        return GateResult(
            passed=False,
            failed_check="non_finite_dropped_fraction",
            detail=(
                f"{n_dropped}/{n_total} rows ({dropped_fraction:.1%}) dropped for a non-finite "
                f"required column; exceeds the {MAX_DROPPED_FRACTION:.0%} cap"
            ),
            n_rows_dropped=n_dropped,
            dropped_fraction=dropped_fraction,
            target_ratio=None,
        )

    max_trained_scale = float(unique_scales.max())
    if target.scale_L <= max_trained_scale:
        return GateResult(
            passed=False,
            failed_check="target_not_interpolation",
            detail=_ERR_INTERPOLATION,
            n_rows_dropped=n_dropped,
            dropped_fraction=dropped_fraction,
            target_ratio=target.scale_L / max_trained_scale,
        )

    target_ratio = target.scale_L / max_trained_scale
    if target_ratio > HARD_CAP_RATIO:
        return GateResult(
            passed=False,
            failed_check="target_ratio_hard_cap",
            detail=(
                f"target scale-up ratio {target_ratio:.2f}x exceeds the {HARD_CAP_RATIO:.0f}x hard cap"
            ),
            n_rows_dropped=n_dropped,
            dropped_fraction=dropped_fraction,
            target_ratio=target_ratio,
        )

    return GateResult(
        passed=True,
        failed_check=None,
        detail="passed",
        n_rows_dropped=n_dropped,
        dropped_fraction=dropped_fraction,
        target_ratio=target_ratio,
        clean_df=clean_df,
    )


# --- ladder ------------------------------------------------------------------ #


@dataclass(frozen=True)
class RungResult:
    """One backtested step: predicting `scale_L` from every smaller scale."""

    scale_L: float
    step_ratio: float
    n: int
    mae: float
    naive_mean_mae: float
    naive_nn_mae: float
    beats_both: bool


@dataclass(frozen=True)
class SkippedRung:
    scale_L: float
    reason: str


@dataclass(frozen=True)
class LadderResult:
    """The nested leave-last-scale-out backtest's full output."""

    rungs: list[RungResult]
    skipped: list[SkippedRung]
    residuals: NDArray[np.float64]  # pooled |actual - pred| across every rung
    pooled_mae: float
    pooled_naive_mean_mae: float
    pooled_naive_nn_mae: float
    pooled_beats_both: bool


def ladder(
    df: pd.DataFrame,
    target_column: str,
    process_columns: Sequence[str],
    bounds: NDArray[np.float64],
    *,
    config: ScaleFeatureConfig = DEFAULT_SCALE_FEATURE_CONFIG,
) -> LadderResult:
    """Nested leave-last-scale-out ladder over `df`'s distinct scales.

    For each scale k (ascending) whose strictly-smaller scales give at least
    `MIN_RUNG_TRAIN_SCALES` distinct scales and at least `MIN_RUNG_TRAIN_ROWS`
    rows, calls `leave_one_scale_out_report` on `df` truncated to scales
    `<= k` with `held_out_scales=[k]`, keeping that fold's residuals and
    baseline comparison. Every rung and the final fit both use the SAME
    `bounds` (an explicit box spanning training AND target - the caller's
    responsibility to build), so the backtest describes the same model the
    final prediction uses (R2-9 in the design's decision ledger).

    Scales ineligible for a rung (too few distinct smaller scales, or too
    few rows to train on) are recorded in `skipped`, never silently dropped.
    """
    scale_col = config.scale_column
    scales = sorted(float(s) for s in np.unique(np.round(df[scale_col].to_numpy(dtype=float), 6)))

    rungs: list[RungResult] = []
    skipped: list[SkippedRung] = []
    all_actual: list[float] = []
    all_pred: list[float] = []
    all_naive_mean: list[float] = []
    all_naive_nn: list[float] = []

    for i in range(1, len(scales)):
        k = scales[i]
        smaller = scales[:i]
        train_mask = df[scale_col].round(6).isin(smaller)
        n_train_scales = len(smaller)
        n_train_rows = int(train_mask.sum())
        if n_train_scales < MIN_RUNG_TRAIN_SCALES or n_train_rows < MIN_RUNG_TRAIN_ROWS:
            skipped.append(
                SkippedRung(
                    scale_L=k,
                    reason=(
                        f"training set has {n_train_scales} distinct scale(s) and {n_train_rows} row(s); "
                        f"need at least {MIN_RUNG_TRAIN_SCALES} scales and {MIN_RUNG_TRAIN_ROWS} rows"
                    ),
                )
            )
            continue

        sub = df.loc[train_mask | df[scale_col].round(6).eq(k)]
        report = leave_one_scale_out_report(
            sub, target_column, process_columns, config=config, bounds=bounds, include_oof=True, held_out_scales=[k]
        )
        if not report["per_scale"]:
            skipped.append(SkippedRung(scale_L=k, reason="fold guard skipped this scale (too few rows)"))
            continue
        row = report["per_scale"][0]
        step_ratio = k / smaller[-1]
        rungs.append(
            RungResult(
                scale_L=k,
                step_ratio=step_ratio,
                n=row["n"],
                mae=row["mae"],
                naive_mean_mae=row["naive_mean_mae"],
                naive_nn_mae=row["naive_nn_mae"],
                beats_both=bool(row["beats_naive_mean"] and row["beats_naive_nn"]),
            )
        )
        oof = report["oof"]
        all_actual.extend(oof["actual"])
        all_pred.extend(oof["pred"])
        all_naive_mean.extend(oof["naive_mean_pred"])
        all_naive_nn.extend(oof["naive_nn_pred"])

    actual_arr = np.asarray(all_actual, dtype=float)
    pred_arr = np.asarray(all_pred, dtype=float)
    naive_mean_arr = np.asarray(all_naive_mean, dtype=float)
    naive_nn_arr = np.asarray(all_naive_nn, dtype=float)

    residuals = np.abs(actual_arr - pred_arr)
    pooled_mae = float(np.mean(residuals)) if len(residuals) else float("nan")
    pooled_naive_mean_mae = float(np.mean(np.abs(actual_arr - naive_mean_arr))) if len(residuals) else float("nan")
    nn_finite = np.isfinite(naive_nn_arr)
    pooled_naive_nn_mae = (
        float(np.mean(np.abs(actual_arr[nn_finite] - naive_nn_arr[nn_finite]))) if nn_finite.any() else float("nan")
    )
    pooled_beats_both = bool(
        np.isfinite(pooled_mae)
        and np.isfinite(pooled_naive_mean_mae)
        and np.isfinite(pooled_naive_nn_mae)
        and pooled_mae < pooled_naive_mean_mae
        and pooled_mae < pooled_naive_nn_mae
    )

    return LadderResult(
        rungs=rungs,
        skipped=skipped,
        residuals=residuals,
        pooled_mae=pooled_mae,
        pooled_naive_mean_mae=pooled_naive_mean_mae,
        pooled_naive_nn_mae=pooled_naive_nn_mae,
        pooled_beats_both=pooled_beats_both,
    )


# --- decision table ----------------------------------------------------------- #

NUMBER = "NUMBER"
NUMBER_WITH_WARNING = "NUMBER_WITH_WARNING"
REFUSAL = "REFUSAL"


@dataclass(frozen=True)
class DecisionResult:
    decision: str  # NUMBER | NUMBER_WITH_WARNING | REFUSAL
    reasons: list[str]
    reference_ratio: float | None
    requested_ratio: float
    out_of_range_params: list[str]


def decide(
    ladder_result: LadderResult,
    requested_ratio: float,
    target_process_params: dict[str, float],
    trained_param_ranges: dict[str, tuple[float, float]],
) -> DecisionResult:
    """The decision table (design doc step 6). Process parameters are
    checked against the trained range separately from the scale axis - the
    scale/agitation/airflow physics inputs are EXPECTED to move outside the
    trained envelope (that is the whole point of scale-up), so only the
    recipe parameters in `target_process_params` drive the warning.
    """
    winning_rungs = [r for r in ladder_result.rungs if r.beats_both]
    if not winning_rungs:
        return DecisionResult(
            decision=REFUSAL,
            reasons=["no rung's own MAE beat both naive baselines"],
            reference_ratio=None,
            requested_ratio=requested_ratio,
            out_of_range_params=[],
        )
    reference_ratio = max(r.step_ratio for r in winning_rungs)

    if not ladder_result.pooled_beats_both:
        return DecisionResult(
            decision=REFUSAL,
            reasons=["pooled ladder MAE does not beat both naive baselines"],
            reference_ratio=reference_ratio,
            requested_ratio=requested_ratio,
            out_of_range_params=[],
        )
    if len(ladder_result.residuals) < MIN_RESIDUALS:
        return DecisionResult(
            decision=REFUSAL,
            reasons=[f"only {len(ladder_result.residuals)} pooled ladder residual(s); need at least {MIN_RESIDUALS}"],
            reference_ratio=reference_ratio,
            requested_ratio=requested_ratio,
            out_of_range_params=[],
        )
    if requested_ratio > REFUSE_RATIO_MULT * reference_ratio:
        return DecisionResult(
            decision=REFUSAL,
            reasons=[
                f"target ratio {requested_ratio:.2f}x exceeds {REFUSE_RATIO_MULT:.0f}x the reference "
                f"step ratio {reference_ratio:.2f}x"
            ],
            reference_ratio=reference_ratio,
            requested_ratio=requested_ratio,
            out_of_range_params=[],
        )

    out_of_range = sorted(
        name
        for name, value in target_process_params.items()
        if name in trained_param_ranges
        and not (trained_param_ranges[name][0] <= value <= trained_param_ranges[name][1])
    )

    reasons: list[str] = []
    warn = False
    if requested_ratio > WARN_RATIO_MULT * reference_ratio:
        warn = True
        reasons.append(
            f"target ratio {requested_ratio:.2f}x exceeds {WARN_RATIO_MULT:.0f}x the reference "
            f"step ratio {reference_ratio:.2f}x"
        )
    if out_of_range:
        warn = True
        reasons.append(f"process parameter(s) outside the trained range: {', '.join(out_of_range)}")

    return DecisionResult(
        decision=NUMBER_WITH_WARNING if warn else NUMBER,
        reasons=reasons,
        reference_ratio=reference_ratio,
        requested_ratio=requested_ratio,
        out_of_range_params=out_of_range,
    )


# --- physics assumption provenance ------------------------------------------- #

_GEOMETRY_FIELDS = {f.name for f in fields(GeometryAssumptions)}
_POWER_FIELDS = {f.name for f in fields(PowerNumberAssumption)}
_KLA_FIELDS = {f.name for f in fields(VantRietParams)}
_OTHER_FIELDS = {"reference_scale_L"}
_ALL_PHYSICS_FIELDS = _GEOMETRY_FIELDS | _POWER_FIELDS | _KLA_FIELDS | _OTHER_FIELDS


def _build_physics_config(
    physics_overrides: dict[str, float] | None,
) -> tuple[ScaleFeatureConfig, dict[str, dict[str, Any]]]:
    """Build a `ScaleFeatureConfig` from `physics_overrides` (a flat dict
    keyed by the dataclass field names in `GeometryAssumptions` /
    `PowerNumberAssumption` / `VantRietParams`, plus `reference_scale_L`),
    and the per-field provenance (`value`, `source`: `"default"` or
    `"user-supplied"`) every assumption is printed with.

    Raises `ValueError` naming any override key that is not one of those
    fields.
    """
    overrides = dict(physics_overrides or {})
    unknown = set(overrides) - _ALL_PHYSICS_FIELDS
    if unknown:
        raise ValueError(f"unknown physics_overrides key(s): {sorted(unknown)}")

    geometry = GeometryAssumptions(**{k: overrides[k] for k in _GEOMETRY_FIELDS if k in overrides})
    power = PowerNumberAssumption(**{k: overrides[k] for k in _POWER_FIELDS if k in overrides})
    kla = VantRietParams(**{k: overrides[k] for k in _KLA_FIELDS if k in overrides})
    reference_scale_L = overrides.get("reference_scale_L", DEFAULT_SCALE_FEATURE_CONFIG.reference_scale_L)

    config = ScaleFeatureConfig(
        scale_column=DEFAULT_SCALE_FEATURE_CONFIG.scale_column,
        agitation_column=DEFAULT_SCALE_FEATURE_CONFIG.agitation_column,
        airflow_column=DEFAULT_SCALE_FEATURE_CONFIG.airflow_column,
        reference_scale_L=reference_scale_L,
        geometry=geometry,
        power=power,
        kla_params=kla,
    )

    assumptions: dict[str, dict[str, Any]] = {}
    for name in sorted(_ALL_PHYSICS_FIELDS):
        value = getattr(geometry, name, None)
        if value is None:
            value = getattr(power, name, None)
        if value is None:
            value = getattr(kla, name, None)
        if name == "reference_scale_L":
            value = reference_scale_L
        assumptions[name] = {"value": value, "source": "user-supplied" if name in overrides else "default"}
    return config, assumptions


# --- normalize plan provenance ----------------------------------------------- #


def _normalize_provenance(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, NormalizationPlan, str, str]:
    """Run `kalos.normalize`'s offline plan/apply flow over the raw upload
    for provenance and unit resolution, and return a working frame with the
    ORIGINAL column names restored.

    `offline_plan`/`apply_plan` canonicalize every kept column's name to
    lowercase snake_case (e.g. `scale_L` -> `scale_l`), which would break
    every literal column-name match `kalos.scale` makes (`scale_L`,
    `agitation_rpm`, `airflow_L_per_min` are exact-cased constants). Rather
    than rename the physics pipeline, this renames the plan's OWN output
    back to each column's raw name, so the pipeline sees the original
    literal names with any genuine unit conversion / numeric coercion the
    plan actually did already applied. The plan JSON and the normalized
    frame's hash (see `build_readout`) are taken from the plan's untouched
    output, matching what a client re-running normalization would get.
    """
    plan = offline_plan(df)
    result = apply_plan(df, plan)
    rename_back = {
        p.canonical_name: p.raw_name
        for p in result.provenance
        if p.canonical_name is not None and p.canonical_name in result.frame.columns
    }
    working = result.frame.rename(columns=rename_back)
    normalized_csv = result.frame.to_csv(index=False)
    return working, plan, normalized_csv, plan.to_json()


def _resolved_unit_for(plan: NormalizationPlan, raw_name: str) -> str:
    """The unit the normalize plan resolved for `raw_name` - a human label
    (e.g. `"rpm"`) if a unit conversion happened, else `"no unit conversion
    applied"`."""
    for col in plan.columns:
        if col.raw_name == raw_name:
            if col.to_base and col.unit_token:
                suffix = canonical_suffix(col.unit_token)
                return base_unit_label(suffix) if suffix else str(col.unit_token)
            return "no unit conversion applied"
    return "column not in plan"


# --- orchestration ------------------------------------------------------------ #


def _git_sha() -> str:
    """kalos git SHA for provenance. No version/commit mechanism is exposed
    anywhere in the portal today (see `kalos/portal/app.py`'s `/healthz`
    docstring), so this falls back to `KALOS_GIT_SHA` if set, else the
    literal `"unknown"` - never fabricated."""
    import os

    return os.environ.get("KALOS_GIT_SHA", "").strip() or "unknown"


def build_readout(
    df: pd.DataFrame,
    target_column: str,
    process_columns: Sequence[str],
    target: TargetSpec,
    *,
    candidate: str = CANDIDATE,
    physics_overrides: dict[str, float] | None = None,
    raw_bytes: bytes | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Orchestrate gate -> ladder -> decide -> final fit -> predict ->
    interval, and return a JSON-serializable dict with every section the
    readout needs (see the module docstring's PIPELINE section).

    `physics_overrides` (optional): a flat dict of `GeometryAssumptions` /
    `PowerNumberAssumption` / `VantRietParams` field names (plus
    `reference_scale_L`) to override; any value not supplied falls back to
    the literature default, and every assumption is printed tagged
    `"default"` or `"user-supplied"` (see `_build_physics_config`). Column
    names (`scale_L`, `agitation_rpm`, `airflow_L_per_min`) always use
    `DEFAULT_SCALE_FEATURE_CONFIG`'s - the input contract fixes them.

    `raw_bytes` (optional): the raw upload bytes, hashed (SHA-256) into
    provenance. `seed` is recorded in provenance for reproducibility, even
    though nothing in this pipeline currently draws randomness (the GP fit
    and the ladder's leave-one-scale-out splits are both deterministic).
    """
    working_config, physics_assumptions = _build_physics_config(physics_overrides)

    working_df, plan, normalized_csv, plan_json = _normalize_provenance(df)

    gate_result = gate(working_df, target_column, process_columns, target, config=working_config)

    provenance: dict[str, Any] = {
        "kalos_git_sha": _git_sha(),
        "candidate": candidate,
        "raw_upload_sha256": hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None,
        "normalized_frame_sha256": hashlib.sha256(normalized_csv.encode("utf-8")).hexdigest(),
        "alpha": ALPHA,
        "seed": seed,
        "normalize_plan_json": plan_json,
        "resolved_scale_unit": _resolved_unit_for(plan, working_config.scale_column),
        "constants": dict(CONSTANTS),
    }

    base: dict[str, Any] = {
        "decision": None,
        "reasons": [],
        "prediction": None,
        "interval": None,
        "interval_label": "approximate coverage",
        "rungs": [],
        "skipped_rungs": [],
        "reference_ratio": None,
        "requested_ratio": gate_result.target_ratio,
        "out_of_range_params": [],
        "baseline_comparison": None,
        "target_inputs": {
            "scale_L": target.scale_L,
            "agitation_rpm": target.agitation_rpm,
            "airflow_L_per_min": target.airflow_L_per_min,
            "process_params": dict(target.process_params),
        },
        "physics_assumptions": physics_assumptions,
        "provenance": provenance,
        "gate": {
            "passed": gate_result.passed,
            "failed_check": gate_result.failed_check,
            "detail": gate_result.detail,
            "n_rows_dropped": gate_result.n_rows_dropped,
            "dropped_fraction": gate_result.dropped_fraction,
        },
    }

    if not gate_result.passed:
        base["decision"] = REFUSAL
        base["reasons"] = [gate_result.detail]
        return base

    assert gate_result.clean_df is not None
    assert gate_result.target_ratio is not None  # always set when gate passes
    clean_df = gate_result.clean_df
    target_ratio = gate_result.target_ratio

    # explicit bounds spanning training AND target - the SAME box for the
    # ladder backtest and the final fit (R2-9 in the design's decision
    # ledger), so the backtested model and the deployed model agree.
    X_train, _ = build_feature_matrix(clean_df, list(process_columns), working_config)
    X_target, _ = build_feature_matrix(target.to_frame(working_config), list(process_columns), working_config)
    bounds = np.vstack(
        [np.minimum(X_train.min(axis=0), X_target.min(axis=0)), np.maximum(X_train.max(axis=0), X_target.max(axis=0))]
    )

    ladder_result = ladder(clean_df, target_column, process_columns, bounds, config=working_config)

    trained_param_ranges = {
        col: (float(clean_df[col].min()), float(clean_df[col].max())) for col in process_columns
    }
    decision_result = decide(ladder_result, target_ratio, target.process_params, trained_param_ranges)

    base["decision"] = decision_result.decision
    base["reasons"] = decision_result.reasons
    base["rungs"] = [r.__dict__ for r in ladder_result.rungs]
    base["skipped_rungs"] = [s.__dict__ for s in ladder_result.skipped]
    base["reference_ratio"] = decision_result.reference_ratio
    base["requested_ratio"] = decision_result.requested_ratio
    base["out_of_range_params"] = decision_result.out_of_range_params
    base["baseline_comparison"] = {
        "pooled_mae": ladder_result.pooled_mae,
        "pooled_naive_mean_mae": ladder_result.pooled_naive_mean_mae,
        "pooled_naive_nn_mae": ladder_result.pooled_naive_nn_mae,
        "pooled_beats_both": ladder_result.pooled_beats_both,
        "n_residuals": int(len(ladder_result.residuals)),
    }

    if decision_result.decision == REFUSAL:
        return base

    model = ScaleUpTransferModel(list(process_columns), target_column, working_config, candidate=candidate)
    model.fit(clean_df, bounds=bounds)
    mean, _std = model.predict(target.to_frame(working_config))
    q = q_from_residuals(ladder_result.residuals, alpha=ALPHA)

    prediction = float(mean[0])
    lower, upper = conformal_interval(mean, q)[0]
    base["prediction"] = prediction
    base["interval"] = [float(lower), float(upper)]
    return base


__all__ = [
    "MIN_SCALES",
    "MIN_RUNS_PER_SCALE",
    "MIN_RUNG_TARGET_ROWS",
    "MIN_RUNG_TRAIN_SCALES",
    "MIN_RUNG_TRAIN_ROWS",
    "MAX_DROPPED_FRACTION",
    "HARD_CAP_RATIO",
    "WARN_RATIO_MULT",
    "REFUSE_RATIO_MULT",
    "ALPHA",
    "MIN_RESIDUALS",
    "CANDIDATE",
    "CONSTANTS",
    "NUMBER",
    "NUMBER_WITH_WARNING",
    "REFUSAL",
    "TargetSpec",
    "GateResult",
    "RungResult",
    "SkippedRung",
    "LadderResult",
    "DecisionResult",
    "gate",
    "ladder",
    "decide",
    "build_readout",
]

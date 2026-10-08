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
  1. `gate` - five named checks, in order; the first one that fails stops
     the pipeline and names itself, so a caller never sees a silent partial
     answer. Real tech-transfer sheets often have many bench runs and only
     one or two runs at each large scale, so the gate does not require a
     minimum run count at every scale, or a minimum row count at the rung
     scales - it only requires enough distinct scales to form a rung at
     all. Whether there is enough EVIDENCE to license a number is decided
     later, by `decide` and `MIN_RESIDUALS`/`MIN_RUNG_N_FOR_LICENSE` below,
     and insufficient evidence is a REFUSAL, not a 422.
  2. `ladder` - a nested leave-last-scale-out backtest: for every scale
     (except the smallest two, which cannot form a >=2-scale training set),
     fit on every smaller scale and predict that scale, keeping the
     per-row residuals and the naive-baseline comparison. This measures the
     model's demonstrated scale-transfer skill, at whatever step ratios the
     uploaded sheet actually lets it test. A rung's held-out scale may have
     any number of runs >= 1; a rung with fewer than `MIN_RUNG_N_FOR_LICENSE`
     runs still contributes its residuals to the pool, but is marked
     `too_few_to_judge` and can never license the reference step ratio.
  3. `decide` - NUMBER / NUMBER_WITH_WARNING / REFUSAL from the ladder's
     evidence. A number is NEVER issued without an interval. Insufficient
     evidence (too few pooled residuals, or no rung that can license a
     reference) is a REFUSAL that still shows everything computed so far,
     plus a "what it would take" data plan.
  4. Final fit (`ScaleUpTransferModel`, candidate `v0`, EXPLICIT bounds
     spanning training AND target) -> `predict` at the target -> a
     split-conformal interval calibrated on the ladder's pooled residuals
     (labeled "approximate coverage": exchangeability does not hold across
     an untested scale, so this is not called calibrated).

UNITS. `scale_L` is liters, `agitation_rpm` is rpm, `airflow_L_per_min` is
liters per minute - the same units `kalos.scale.features` assumes.

SCALE-ONLY FALLBACK (owner decision 2026-09-24). Real run sheets often do
not record `agitation_rpm`/`airflow_L_per_min`. `build_readout` reads the
uploaded sheet's own columns to choose a `feature_set` (see
`kalos.scale.transfer.ScaleFeatureConfig.feature_set`): `"physics"` when
both columns are present (unchanged behavior), `"scale_only"` when either
is absent, using only `kalos.scale.features.SCALE_ONLY_FEATURE_COLUMNS`
(`log_volume_ratio`, `hydrostatic_pressure_mmHg` - the physics features
computable from `scale_L` alone). This is a deliberate FEATURE-SET
choice, not an imputation: a missing column silently turned into an
all-NaN physics feature would make `ScaleUpTransferModel.fit`'s
no-imputation contract drop every row. `TargetSpec.agitation_rpm`/
`airflow_L_per_min` are required only in `"physics"` mode (`gate` raises
`ValueError` naming whichever is missing); in `"scale_only"` mode they are
optional, and a caller-supplied value is ignored (recorded as such) since
the sheet gives the model nothing to relate it to. `feature_set` and
`missing_physics_inputs` are carried in the returned dict and its
`provenance`, and the physics assumptions section only reports the
assumptions the selected feature set actually uses.

Every threshold below is a named module-level constant so the readout can
print exactly what gated it, rather than a caller having to reverse-engineer
a number baked into a comparison.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kalos import __version__ as ENGINE_VERSION
from kalos.core.conformal import conformal_interval, q_from_residuals
from kalos.normalize import NormalizationPlan, apply_plan, offline_plan
from kalos.normalize.units import base_unit_label, canonical_suffix

from .evaluation import leave_one_scale_out_report
from .features import GeometryAssumptions, PowerNumberAssumption, VantRietParams
from .transfer import DEFAULT_SCALE_FEATURE_CONFIG, ScaleFeatureConfig, ScaleUpTransferModel, build_feature_matrix

# --- named constants (every one of these is printed in the readout's ----- #
# --- provenance section - see Code Quality review finding 1). ----------- #
MIN_SCALES = 3
MIN_RUNG_TRAIN_SCALES = 2
MIN_RUNG_TRAIN_ROWS = 4
MIN_RUNG_N_FOR_LICENSE = 3
# A rung can license the reference step ratio only if its held-out scale is
# within this many decades of the largest trained scale. A 5x step won at
# 0.05 -> 0.25 L says little about a 5x step at plant scale, where mixing and
# oxygen transfer behave differently; ratio alone ignores the scale regime.
LICENSE_WINDOW_DECADES = 1.0
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
    "MIN_RUNG_TRAIN_SCALES": MIN_RUNG_TRAIN_SCALES,
    "MIN_RUNG_TRAIN_ROWS": MIN_RUNG_TRAIN_ROWS,
    "MIN_RUNG_N_FOR_LICENSE": MIN_RUNG_N_FOR_LICENSE,
    "LICENSE_WINDOW_DECADES": LICENSE_WINDOW_DECADES,
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
    `process_params`: the recipe inputs to be run at the target scale, one
    entry per column in `process_columns` (e.g. `{"ph_setpoint": 7.0,
    "temperature_C": 37.0}`) - checked against the trained range separately
    from the scale axis (see `decide`).
    `agitation_rpm`: planned impeller speed at the target scale, rpm.
    `airflow_L_per_min`: planned sparge/gas flow at the target scale, liters
    per minute.

    `agitation_rpm`/`airflow_L_per_min` are OPTIONAL (default `None`):
    `build_readout` selects the "physics" feature set only when the
    uploaded sheet records BOTH `agitation_rpm` and `airflow_L_per_min`
    columns, and `gate` then raises `ValueError` naming whichever of these
    two is `None` - a planned physics feature set with no planned value to
    feed it is a caller/target-JSON error, not a gate failure. When the
    sheet is missing either column, `build_readout` falls back to the
    "scale_only" feature set and these two fields become genuinely
    optional; a value supplied anyway is recorded as ignored (the sheet
    does not record what it would take to use it).
    """

    scale_L: float
    process_params: dict[str, float]
    agitation_rpm: float | None = None
    airflow_L_per_min: float | None = None

    def to_frame(self, config: ScaleFeatureConfig) -> pd.DataFrame:
        """A single-row frame in the shape `build_feature_matrix` expects.

        `agitation_rpm`/`airflow_L_per_min` are only written onto the row
        when not `None`. In "scale_only" mode `build_feature_matrix` never
        reads those columns anyway (see `SCALE_ONLY_FEATURE_COLUMNS`), so
        omitting them keeps this frame honest about what was actually
        supplied rather than writing a value that is silently unused.
        """
        row = dict(self.process_params)
        row[config.scale_column] = self.scale_L
        if self.agitation_rpm is not None:
            row[config.agitation_column] = self.agitation_rpm
        if self.airflow_L_per_min is not None:
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
    """`scale_L`, the process columns, and the target column.

    `agitation_rpm`/`airflow_L_per_min` are deliberately NOT required
    here: `build_readout` reads them when present to pick the "physics"
    feature set (see `kalos.scale.transfer.ScaleFeatureConfig.feature_set`)
    but falls back to "scale_only" when either is absent, rather than
    422-ing a real run sheet that simply does not record them."""
    return [target_column, config.scale_column, *process_columns]


def gate(
    df: pd.DataFrame,
    target_column: str,
    process_columns: Sequence[str],
    target: TargetSpec,
    *,
    config: ScaleFeatureConfig = DEFAULT_SCALE_FEATURE_CONFIG,
) -> GateResult:
    """Five named checks, in order; the first failure stops the pipeline.

    1. `required_columns` - every required column is present.
    2. `min_distinct_scales` - at least `MIN_SCALES` distinct scales are
       present. This is a structural minimum, not an evidence minimum: with
       fewer than `MIN_SCALES` scales no rung can ever be formed, no matter
       how many runs exist at each one. There is deliberately no minimum run
       count per scale and no minimum row count at the rung-eligible
       scales - real tech-transfer sheets routinely have dozens of bench
       runs and only one or two runs at each large scale, and that shape is
       exactly what this readout exists to read honestly. Whether the
       resulting evidence is enough to license a number is `decide`'s job,
       not the gate's; too little evidence is a REFUSAL, not a 422.
    3. `non_finite_dropped_fraction` - the fraction of rows dropped for a
       non-finite required column is at most `MAX_DROPPED_FRACTION`.
    4. `target_not_interpolation` - the target scale is strictly larger than
       every trained scale.
    5. `target_ratio_hard_cap` - the target/largest-trained-scale ratio is
       at most `HARD_CAP_RATIO`.

    If `target.process_params`' keys do not exactly match `process_columns`,
    raises `ValueError` (a caller/target-JSON error, not a gate failure - the
    portal route maps this to its own 422 before ever calling `gate`).

    Likewise raises `ValueError`, naming whichever is missing, if
    `config.feature_set == "physics"` (selected by the caller - see
    `build_readout`'s feature_set selection, based on which columns the
    uploaded sheet actually has) and `target.agitation_rpm` and/or
    `target.airflow_L_per_min` is `None`: a planned physics feature set
    with no planned value to feed it is the same kind of caller/target-JSON
    error as a `process_params` mismatch, not a gate failure.
    """
    if set(target.process_params) != set(process_columns):
        raise ValueError(
            f"target.process_params keys {sorted(target.process_params)} do not match "
            f"process_columns {sorted(process_columns)}"
        )

    if config.feature_set == "physics":
        missing_target_fields = [
            name
            for name, value in (
                ("agitation_rpm", target.agitation_rpm),
                ("airflow_L_per_min", target.airflow_L_per_min),
            )
            if value is None
        ]
        if missing_target_fields:
            raise ValueError(
                "target must supply " + ", ".join(missing_target_fields) + ": this sheet records both "
                "agitation_rpm and airflow_L_per_min, so the physics feature set needs a planned "
                "value for each"
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
        needed_more = MIN_SCALES - len(unique_scales)
        return GateResult(
            passed=False,
            failed_check="min_distinct_scales",
            detail=(
                f"only {len(unique_scales)} distinct scale(s) present; need {needed_more} more "
                f"(at least {MIN_SCALES} distinct scales total)"
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
    """One backtested step: predicting `scale_L` from every smaller scale.

    `too_few_to_judge` (n < `MIN_RUNG_N_FOR_LICENSE`): the held-out scale's
    residuals still pool into the ladder's evidence, but this rung's own
    "beats both" verdict is too thin a sample to trust, and it can never
    license the reference step ratio (see `decide`).
    """

    scale_L: float
    step_ratio: float
    n: int
    mae: float
    naive_mean_mae: float
    naive_nn_mae: float
    beats_both: bool
    too_few_to_judge: bool
    # held-out scale is within LICENSE_WINDOW_DECADES of the largest trained scale
    in_license_window: bool = True


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
        in_window = bool(np.log10(scales[-1] / k) <= LICENSE_WINDOW_DECADES + 1e-9)
        rungs.append(
            RungResult(
                scale_L=k,
                step_ratio=step_ratio,
                n=row["n"],
                mae=row["mae"],
                naive_mean_mae=row["naive_mean_mae"],
                naive_nn_mae=row["naive_nn_mae"],
                beats_both=bool(row["beats_naive_mean"] and row["beats_naive_nn"]),
                too_few_to_judge=row["n"] < MIN_RUNG_N_FOR_LICENSE,
                in_license_window=in_window,
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


def _can_license(rung: RungResult) -> bool:
    """A rung licenses the reference step ratio only when it beat both
    baselines, on enough runs, near the scale regime being extrapolated from."""
    return rung.beats_both and not rung.too_few_to_judge and rung.in_license_window


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
    # Every refusal condition that applies is reported, not just the first:
    # a prospect reading a refusal needs the whole list to plan more runs.
    refusal_reasons: list[str] = []
    n_residuals = len(ladder_result.residuals)
    if n_residuals < MIN_RESIDUALS:
        refusal_reasons.append(f"only {n_residuals} pooled ladder residual(s); need at least {MIN_RESIDUALS}")

    # a rung with fewer than MIN_RUNG_N_FOR_LICENSE runs can beat both
    # baselines by chance on that thin a sample, so it is never allowed to
    # license the reference step ratio (see RungResult.too_few_to_judge).
    licensing_rungs = [r for r in ladder_result.rungs if _can_license(r)]
    reference_ratio = max((r.step_ratio for r in licensing_rungs), default=None)
    if reference_ratio is None:
        refusal_reasons.append(
            f"no rung with at least {MIN_RUNG_N_FOR_LICENSE} runs, within {LICENSE_WINDOW_DECADES:g} decade(s) "
            "of the largest trained scale, beat both naive baselines to license a reference step ratio"
        )

    # the pooled verdict is only a verdict with enough residuals behind it
    if n_residuals >= MIN_RESIDUALS and not ladder_result.pooled_beats_both:
        refusal_reasons.append("pooled ladder MAE does not beat both naive baselines")

    if reference_ratio is not None and requested_ratio > REFUSE_RATIO_MULT * reference_ratio:
        refusal_reasons.append(
            f"target ratio {requested_ratio:.2f}x exceeds {REFUSE_RATIO_MULT:.0f}x the reference "
            f"step ratio {reference_ratio:.2f}x"
        )

    if refusal_reasons:
        return DecisionResult(
            decision=REFUSAL,
            reasons=refusal_reasons,
            reference_ratio=reference_ratio,
            requested_ratio=requested_ratio,
            out_of_range_params=[],
        )
    assert reference_ratio is not None  # no refusal reasons implies a licensing rung

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


# --- "what it would take" data plan ------------------------------------------ #


def format_liters_plain(value: float) -> str:
    """Up to 4 significant digits, never scientific notation - the same
    convention `kalos.portal.scale_routes._fmt_sig4` renders every scale
    value with (that module delegates to this function so a data-plan
    liters figure and a rendered scale figure never disagree). Magnitudes
    of 10^4 and above print as grouped integers ("40,000"); smaller values
    keep `{:.4g}` ("0.3333", "7.2", "1000")."""
    v = float(value)
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def _smallest_rung_target_scale(clean_df: pd.DataFrame, config: ScaleFeatureConfig) -> float:
    """The smallest scale that can be a rung target: the first scale
    (ascending) whose strictly-smaller scales satisfy the rung training
    minimums (`MIN_RUNG_TRAIN_SCALES` distinct scales, `MIN_RUNG_TRAIN_ROWS`
    rows) - in practice almost always the third-smallest distinct scale,
    since that is usually enough rows. Falls back to the third-smallest
    scale if none qualifies among the scales actually present (`gate`
    guarantees at least `MIN_SCALES` distinct scales, so index 2 exists)."""
    scale_col = config.scale_column
    scales = sorted(float(s) for s in np.unique(np.round(clean_df[scale_col].to_numpy(dtype=float), 6)))
    for i in range(1, len(scales)):
        smaller = scales[:i]
        if len(smaller) < MIN_RUNG_TRAIN_SCALES:
            continue
        train_mask = clean_df[scale_col].round(6).isin(smaller)
        if int(train_mask.sum()) >= MIN_RUNG_TRAIN_ROWS:
            return scales[i]
    return scales[2]


def _build_data_plan(
    ladder_result: LadderResult,
    decision_result: DecisionResult,
    requested_scale_L: float,
    clean_df: pd.DataFrame,
    config: ScaleFeatureConfig,
) -> list[str]:
    """Plain, specific "what it would take" lines, computed straight from
    the same rules `decide` applies - never an invented statistic. Each
    condition below is checked independently of which reason `decide`
    itself returned first, so a REFUSAL page shows every gap that applies,
    not just the one that happened to short-circuit `decide`."""
    plan: list[str] = []
    n_residuals = len(ladder_result.residuals)
    x_scale = _smallest_rung_target_scale(clean_df, config)
    largest_trained_L = float(clean_df[config.scale_column].max())
    license_floor_L = max(x_scale, largest_trained_L / 10**LICENSE_WINDOW_DECADES)

    if n_residuals < MIN_RESIDUALS:
        shortfall = MIN_RESIDUALS - n_residuals
        plan.append(f"{shortfall} more run(s) at any scale of {format_liters_plain(x_scale)} L or larger")
        plan.append("runs at the largest scales also build the evidence that licenses larger steps")

    licensing_rungs = [r for r in ladder_result.rungs if _can_license(r)]
    if not licensing_rungs:
        plan.append(
            f"at least {MIN_RUNG_N_FOR_LICENSE} runs at a single scale of {format_liters_plain(license_floor_L)} L or larger "
            "whose backtest beats both baselines"
        )

    reference = decision_result.reference_ratio
    if reference is not None and decision_result.requested_ratio > WARN_RATIO_MULT * reference:
        clean_target_L = requested_scale_L / (WARN_RATIO_MULT * reference)
        line = f"for a clean prediction, add runs at {format_liters_plain(clean_target_L)} L or larger"
        # the "avoid refusal" bar only means something when the ratio is what
        # refused; on a warning the readout already cleared it.
        if decision_result.requested_ratio > REFUSE_RATIO_MULT * reference:
            refusal_target_L = requested_scale_L / (REFUSE_RATIO_MULT * reference)
            line += f"; to avoid refusal, {format_liters_plain(refusal_target_L)} L or larger"
        plan.append(line + " (assuming the reference step holds)")
    return plan


# --- physics assumption provenance ------------------------------------------- #

_GEOMETRY_FIELDS = {f.name for f in fields(GeometryAssumptions)}
_POWER_FIELDS = {f.name for f in fields(PowerNumberAssumption)}
_KLA_FIELDS = {f.name for f in fields(VantRietParams)}
_OTHER_FIELDS = {"reference_scale_L"}
_ALL_PHYSICS_FIELDS = _GEOMETRY_FIELDS | _POWER_FIELDS | _KLA_FIELDS | _OTHER_FIELDS

# The assumptions actually used by the scale_only feature set
# (`log_volume_ratio`, `hydrostatic_pressure_mmHg` - see
# `SCALE_ONLY_FEATURE_COLUMNS`): `reference_scale_L` (log_volume_ratio's
# zero point) plus the geometry fields hydrostatic pressure's liquid-height
# calculation needs (`aspect_ratio_h_over_t` via `tank_geometry_proxy`,
# `liquid_density_kg_per_m3` for `rho * g * h`). NOT
# `impeller_to_tank_diameter_ratio` (only feeds impeller diameter, which
# only `specific_power_w_per_m3` uses) and NOT the power-number/van't Riet
# constants (those feed `specific_power_w_per_m3`/`kla_proxy_per_s`, both
# excluded from `SCALE_ONLY_FEATURE_COLUMNS`).
_SCALE_ONLY_USED_FIELDS = {"aspect_ratio_h_over_t", "liquid_density_kg_per_m3", "reference_scale_L"}
_NOT_USED_SCALE_ONLY = "not used (scale-only)"


def _build_physics_config(
    physics_overrides: dict[str, float] | None,
    *,
    feature_set: Literal["physics", "scale_only"] = "physics",
) -> tuple[ScaleFeatureConfig, dict[str, dict[str, Any]]]:
    """Build a `ScaleFeatureConfig` from `physics_overrides` (a flat dict
    keyed by the dataclass field names in `GeometryAssumptions` /
    `PowerNumberAssumption` / `VantRietParams`, plus `reference_scale_L`),
    and the per-field provenance (`value`, `source`: `"default"`,
    `"user-supplied"`, or - only in `"scale_only"` mode, for a field that
    mode's features do not read - `"not used (scale-only)"`) every
    assumption is printed with.

    `feature_set` (default `"physics"`): set on the returned
    `ScaleFeatureConfig.feature_set` (so every downstream consumer -
    `gate`, `ladder`, `build_feature_matrix` - agrees on which physics
    columns to build), and used here only to tag `_SCALE_ONLY_USED_FIELDS`
    (the assumptions `log_volume_ratio`/`hydrostatic_pressure_mmHg`
    actually read) as `"default"`/`"user-supplied"` same as always, while
    every OTHER assumption is tagged `"not used (scale-only)"` - the
    physics assumptions section should only claim to use what the
    selected feature set actually computes.

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
        feature_set=feature_set,
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
        if feature_set == "scale_only" and name not in _SCALE_ONLY_USED_FIELDS:
            source = _NOT_USED_SCALE_ONLY
        else:
            source = "user-supplied" if name in overrides else "default"
        assumptions[name] = {"value": value, "source": source}
    return config, assumptions


# --- normalize plan provenance ----------------------------------------------- #


def _normalize_provenance(
    df: pd.DataFrame,
    target_column: str,
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
    plan = offline_plan(df, target_column=target_column)
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


def _unit_label_or_none(plan: NormalizationPlan, raw_name: str) -> str | None:
    """The plain unit label the normalize plan resolved for `raw_name`
    (e.g. `"g/L"`), or `None` if no unit conversion happened for that
    column - used to name the prediction/interval rows with the target
    column's unit when one is actually resolvable, and just the column
    name otherwise."""
    for col in plan.columns:
        if col.raw_name == raw_name and col.to_base and col.unit_token:
            suffix = canonical_suffix(col.unit_token)
            return base_unit_label(suffix) if suffix else str(col.unit_token)
    return None


# --- orchestration ------------------------------------------------------------ #


def _utc_now_iso() -> str:
    """Current UTC time, ISO 8601, seconds precision (no microseconds - a
    render timestamp does not need sub-second precision, and dropping it
    keeps every rendered number on the page decimal-free)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_sha() -> str:
    """kalos git SHA for provenance.

    Fallback order: env `KALOS_GIT_SHA` if set; else `git rev-parse --short
    HEAD` run in this package's repo directory (a 2 second timeout, every
    error swallowed - a missing `git` binary, no `.git` directory in a
    deploy image, or a slow filesystem must never fail a readout); else the
    literal `"unknown"` - never fabricated.
    """
    import os

    env_sha = os.environ.get("KALOS_GIT_SHA", "").strip()
    if env_sha:
        return env_sha

    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else "unknown"


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
    `"default"`, `"user-supplied"`, or (scale_only mode, for an assumption
    that mode's features do not read) `"not used (scale-only)"` (see
    `_build_physics_config`). Column names (`scale_L`, `agitation_rpm`,
    `airflow_L_per_min`) always use `DEFAULT_SCALE_FEATURE_CONFIG`'s - the
    input contract fixes them.

    FEATURE SET SELECTION (the scale-only fallback). Real run sheets often
    do not record `agitation_rpm`/`airflow_L_per_min`. This function reads
    the normalized upload's own columns to choose: `"physics"` when BOTH
    are present (unchanged behavior - `target.agitation_rpm` and
    `target.airflow_L_per_min` must then both be supplied, or `gate` raises
    `ValueError` naming whichever is missing); `"scale_only"` when EITHER
    is absent, using only the physics features computable from `scale_L`
    alone (`kalos.scale.features.SCALE_ONLY_FEATURE_COLUMNS`) - never a
    silently all-NaN physics feature, which the no-imputation `fit`
    contract would otherwise turn into "every row dropped". The chosen
    `feature_set` and the list of absent columns (`missing_physics_inputs`)
    are returned at the top level and in `provenance`.

    `raw_bytes` (optional): the raw upload bytes, hashed (SHA-256) into
    provenance. `seed` is recorded in provenance for reproducibility, even
    though nothing in this pipeline currently draws randomness (the GP fit
    and the ladder's leave-one-scale-out splits are both deterministic).
    """
    working_df, plan, normalized_csv, plan_json = _normalize_provenance(df, target_column)

    missing_physics_inputs = [
        name
        for name, col in (
            ("agitation_rpm", DEFAULT_SCALE_FEATURE_CONFIG.agitation_column),
            ("airflow_L_per_min", DEFAULT_SCALE_FEATURE_CONFIG.airflow_column),
        )
        if col not in working_df.columns
    ]
    feature_set: Literal["physics", "scale_only"] = "scale_only" if missing_physics_inputs else "physics"

    working_config, physics_assumptions = _build_physics_config(physics_overrides, feature_set=feature_set)

    gate_result = gate(working_df, target_column, process_columns, target, config=working_config)

    provenance: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "kalos_git_sha": _git_sha(),
        "candidate": candidate,
        "raw_upload_sha256": hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None,
        "normalized_frame_sha256": hashlib.sha256(normalized_csv.encode("utf-8")).hexdigest(),
        "alpha": ALPHA,
        "seed": seed,
        "normalize_plan_json": plan_json,
        "normalize_plan_sha256": hashlib.sha256(plan_json.encode("utf-8")).hexdigest(),
        "normalize_plan_n_columns": len(plan.columns),
        "resolved_scale_unit": _resolved_unit_for(plan, working_config.scale_column),
        "target_column_unit": _unit_label_or_none(plan, target_column),
        "feature_set": feature_set,
        "missing_physics_inputs": missing_physics_inputs,
        "constants": dict(CONSTANTS),
    }

    base: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "target_column": target_column,
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
        "data_plan": [],
        "feature_set": feature_set,
        "missing_physics_inputs": missing_physics_inputs,
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

    if decision_result.decision in (REFUSAL, NUMBER_WITH_WARNING):
        base["data_plan"] = _build_data_plan(ladder_result, decision_result, target.scale_L, clean_df, working_config)

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
    "MIN_RUNG_TRAIN_SCALES",
    "MIN_RUNG_TRAIN_ROWS",
    "MIN_RUNG_N_FOR_LICENSE",
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
    "format_liters_plain",
]

"""Synthetic-only prototype: does a binary missing-value indicator column help
a GP surrogate's predictive skill, and in which regime?

Kalos's production surrogate (`kalos.core.surrogate.Surrogate`) zero-fills
missing feature values. That is the CORRECT thing to do when "missing" means
"this component is absent from the recipe, so its true contribution is zero" -
which is kalos's stated domain assumption for media components. This script
does NOT touch that assumption or the production code path. It asks a narrower
question on fully synthetic data with a known ground truth: if missingness
instead carried information (Missing Not At Random - MNAR), would adding a
binary `x_k_missing` indicator column recover any of that signal? And in the
domain-matching case (missing really does mean absent), does adding the
indicator cost anything?

Two `missing_mode`s control the simulated ground truth for one feature `x_k`:
  - "absent": missing rows really do have x_k contributing 0 to y (kalos's
    assumption). Zero-fill is exactly correct here BY CONSTRUCTION.
  - "informative": missingness is MNAR - it happens exactly when the true
    (unobserved) x_k is systematically shifted away from 0, and y is generated
    from that real value. Zero-fill silently mis-imputes those rows to x_k=0,
    which is wrong here BY CONSTRUCTION.

Two model arms (same model class, same folds, same everything else):
  - Arm A: zero-fill NaN -> features only (current kalos behavior).
  - Arm B: zero-fill NaN -> features PLUS a binary `x_k_missing` column.

Model choice: a scikit-learn `GaussianProcessRegressor` with an ARD Matern
kernel, NOT `kalos.core.surrogate.Surrogate`. Both are GPs; sklearn's is used
here purely for speed - this script fits ~1200 independent GPs (n_seeds x
p_miss values x missing_modes x arms x folds) and needs to run in about a
minute, and it avoids pulling in torch/botorch/its Cholesky-jitter retry ladder
for a research script that never touches the production path. The comparison
is fair because arm A and arm B always use the identical model class,
hyperparameter search, and CV folds - only the feature matrix differs.

Evaluation is out-of-fold (OOF) Spearman rank correlation between predicted
and true y, pooled across a K-fold CV split, repeated over several seeds and
reported as mean +/- std across seeds. This mirrors the spirit of kalos's own
`grouped_cv_report` (pooled OOF Spearman) without depending on it, since that
helper is wired to the production `Surrogate`.
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# The L-BFGS-B kernel hyperparameter search occasionally lands on a bound (e.g.
# the indicator column's length scale, which is genuinely near-degenerate for a
# binary feature). Sklearn still returns the best log-marginal-likelihood value
# found, so this is benign and expected here, not a correctness problem - it
# just makes ~1200 fits noisy to read on stdout.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_FEATURES = 5  # numeric features in the synthetic recipe
K_INDEX = 2  # the feature subjected to missingness (x_k)
MISSING_MODES = ("absent", "informative")


def true_f(x: np.ndarray) -> np.ndarray:
    """Smooth ground-truth response: a couple of nonlinear terms plus one
    interaction involving x_k (x2), so x_k's contribution is neither trivial
    nor dominant."""
    x0, x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    return 2.0 * np.sin(2.0 * x0) + 1.5 * x1**2 + 2.0 * x2 * x3 - 1.0 * x4


def generate_dataset(
    rng: np.random.Generator, n: int, p_miss: float, missing_mode: str, noise_sd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one synthetic recipe table. Returns (X_observed with NaNs, y, miss_mask).

    `X_observed` has NaN in column K_INDEX for the rows selected as missing.
    y is always generated from the TRUE (possibly unobserved) value of x_k, so
    the ground truth is fully controlled by `missing_mode`.
    """
    x = rng.uniform(-2.0, 2.0, size=(n, N_FEATURES))
    miss_mask = rng.random(n) < p_miss
    n_miss = int(miss_mask.sum())

    if missing_mode == "absent":
        # Domain truth (kalos's assumption): missing = component absent =
        # zero real contribution. Overwrite the true x_k for missing rows
        # with 0 before computing y - zero-fill will exactly match this.
        x[miss_mask, K_INDEX] = 0.0
    elif missing_mode == "informative":
        # Domain truth (MNAR): missingness happens exactly when the real x_k
        # is systematically large and positive - a latent cause zero-fill
        # cannot see. y is generated from this real, nonzero value.
        x[miss_mask, K_INDEX] = rng.uniform(1.0, 3.0, size=n_miss)
    else:
        raise ValueError(f"unknown missing_mode: {missing_mode!r}")

    y = true_f(x) + rng.normal(0.0, noise_sd, size=n)

    x_observed = x.copy()
    x_observed[miss_mask, K_INDEX] = np.nan
    return x_observed, y, miss_mask


def build_features(x_observed: np.ndarray, miss_mask: np.ndarray, *, with_indicator: bool) -> np.ndarray:
    """Arm A: zero-fill only. Arm B: zero-fill plus a binary indicator column."""
    x_filled = np.where(np.isnan(x_observed), 0.0, x_observed)
    if not with_indicator:
        return x_filled
    return np.concatenate([x_filled, miss_mask.astype(float).reshape(-1, 1)], axis=1)


def make_gp(n_features: int, seed: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-6, 1e1))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)


def oof_spearman(x: np.ndarray, y: np.ndarray, *, n_splits: int, seed: int) -> float:
    """Pooled out-of-fold Spearman rho of a GP fit on `x` -> `y`, K-fold CV."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for train_idx, test_idx in kf.split(x):
        scaler = StandardScaler().fit(x[train_idx])
        x_train = scaler.transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        gp = make_gp(x.shape[1], seed)
        gp.fit(x_train, y[train_idx])
        pred[test_idx] = gp.predict(x_test)
    if np.std(pred) == 0 or np.std(y) == 0:
        return float("nan")
    rho = spearmanr(pred, y).statistic
    return float(rho) if rho == rho else float("nan")


@dataclass
class ConfigResult:
    missing_mode: str
    p_miss: float
    arm_a: list[float] = field(default_factory=list)
    arm_b: list[float] = field(default_factory=list)

    @property
    def mean_a(self) -> float:
        return float(np.mean(self.arm_a))

    @property
    def std_a(self) -> float:
        return float(np.std(self.arm_a))

    @property
    def mean_b(self) -> float:
        return float(np.mean(self.arm_b))

    @property
    def std_b(self) -> float:
        return float(np.std(self.arm_b))

    @property
    def mean_delta(self) -> float:
        # Paired per-seed delta (arm B - arm A on the SAME data/folds), then
        # averaged - more honest than diffing the two means separately.
        return float(np.mean(np.array(self.arm_b) - np.array(self.arm_a)))

    @property
    def std_delta(self) -> float:
        return float(np.std(np.array(self.arm_b) - np.array(self.arm_a)))


def run_sweep(n: int, n_seeds: int, p_miss_values: list[float], noise_sd: float, n_splits: int, seed_base: int) -> list[ConfigResult]:
    results = []
    for missing_mode in MISSING_MODES:
        for p_miss in p_miss_values:
            cfg = ConfigResult(missing_mode=missing_mode, p_miss=p_miss)
            for i in range(n_seeds):
                seed = seed_base + i
                rng = np.random.default_rng(seed)
                x_observed, y, miss_mask = generate_dataset(rng, n, p_miss, missing_mode, noise_sd)
                x_a = build_features(x_observed, miss_mask, with_indicator=False)
                x_b = build_features(x_observed, miss_mask, with_indicator=True)
                cfg.arm_a.append(oof_spearman(x_a, y, n_splits=n_splits, seed=seed))
                cfg.arm_b.append(oof_spearman(x_b, y, n_splits=n_splits, seed=seed))
            results.append(cfg)
    return results


def format_table(results: list[ConfigResult]) -> str:
    header = f"{'mode':<12}{'p_miss':>8}{'Arm A (mean+-std)':>22}{'Arm B (mean+-std)':>22}{'delta (B-A)':>18}"
    lines = [header, "-" * len(header)]
    for r in results:
        a = f"{r.mean_a:.3f} +/- {r.std_a:.3f}"
        b = f"{r.mean_b:.3f} +/- {r.std_b:.3f}"
        d = f"{r.mean_delta:+.3f} +/- {r.std_delta:.3f}"
        lines.append(f"{r.missing_mode:<12}{r.p_miss:>8.2f}{a:>22}{b:>22}{d:>18}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="recipes per synthetic dataset")
    parser.add_argument("--n-seeds", type=int, default=10, help="seeds to average over per config")
    parser.add_argument("--p-miss", type=str, default="0.1,0.3,0.5", help="comma-separated missing fractions")
    parser.add_argument("--noise-sd", type=float, default=1.0, help="gaussian noise sd added to y")
    parser.add_argument("--n-splits", type=int, default=5, help="K-fold CV splits")
    parser.add_argument("--seed-base", type=int, default=0, help="base seed; seeds are seed_base..seed_base+n_seeds-1")
    args = parser.parse_args()

    p_miss_values = [float(v) for v in args.p_miss.split(",")]

    t0 = time.time()
    results = run_sweep(
        n=args.n,
        n_seeds=args.n_seeds,
        p_miss_values=p_miss_values,
        noise_sd=args.noise_sd,
        n_splits=args.n_splits,
        seed_base=args.seed_base,
    )
    elapsed = time.time() - t0

    print(f"n={args.n} n_seeds={args.n_seeds} noise_sd={args.noise_sd} n_splits={args.n_splits} "
          f"p_miss={p_miss_values} seed_base={args.seed_base}")
    print(f"elapsed: {elapsed:.1f}s\n")
    print(format_table(results))


if __name__ == "__main__":
    main()

"""Closed-loop optimization benchmark: does `propose` actually find the optimum?

Runs the full pipeline in a loop against the mechanistic simulator (the known
ground truth): train on the runs measured so far, `propose` the next batch, "run"
it in silico, measure titer, append, repeat. Compares the BO/`propose` strategy
against a random-selection baseline by best-found titer per round (regret to the
true optimum from `sim.true_optimum`).

This is an in-silico validation of the acquisition layer, not a client-facing
result - the simulator is a benchmark surface, not a real process.

Run:  cd experiments/leadgene_pipeline && python -m examples.closed_loop_benchmark
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from pipeline import artifact, sim
from pipeline.pipeline import PredictPipeline, TrainPipeline
from pipeline.propose import propose_batch

MODELS = ["gaussian_process", "bayesian_ridge"]   # fast + validate on clean design->titer


def _config(ws: Path) -> dict:
    return {
        "seed": 0,
        "artifact_path": str(ws / "model.joblib"),
        "data": {"train_dir": str(ws / "train"),
                 "predict_csv": str(ws / "candidates.csv"),
                 "output_csv": str(ws / "out" / "predictions.csv")},
        "columns": {"target": "titer", "id_col": "well_id", "group_col": "clone"},
        "features": {"exclude": []},
        "models": MODELS,
        "reference_model": "bayesian_ridge",
        "outputs": {"analysis": False, "visualize": False},
        "preprocess": {"max_numeric_features": 4},
        "cv": {"n_boot": 200},
    }


def _propose_from_pool(cfg: dict, measured: list[tuple[sim.ProcessParams, float]],
                       cand: list[tuple[int, sim.ProcessParams]], q: int, beta: float) -> list[int]:
    """Train on measured (params, titer) pairs, score the unmeasured candidates, and
    return the pool indices of the selected batch."""
    train_dir = Path(cfg["data"]["train_dir"])
    train_dir.mkdir(parents=True, exist_ok=True)
    sim.to_featurized_rows([p for p, _ in measured], [t for _, t in measured]).to_csv(
        train_dir / "measured.csv", index=False)

    methods, feature_cols, reference = TrainPipeline(cfg).run()
    TrainPipeline(cfg).save(cfg["artifact_path"], methods, feature_cols, reference)

    cand_df = sim.to_featurized_rows([p for _, p in cand], [0.0] * len(cand)).drop(columns=["titer"])
    cand_df["well_id"] = [f"c{idx}" for idx, _ in cand]      # well_id encodes the pool index
    cand_df.to_csv(cfg["data"]["predict_csv"], index=False)

    report = PredictPipeline(artifact.load(cfg["artifact_path"]), cfg).run()
    result = propose_batch(report, q=q, beta=beta)
    picked = result.table[result.table["selected"]]["well_id"].tolist()
    return [int(w[1:]) for w in picked]


def run(strategy: str, *, pool_titers: list[float], pool_params: list[sim.ProcessParams],
        rounds: int, n_init: int, q: int, beta: float, seed: int) -> tuple[list[float], list[float]]:
    """One campaign over a FIXED candidate library (both strategies draw from the same
    pool without replacement). Returns (best-found per round, mean titer of the batch
    each round selected)."""
    rng = np.random.default_rng(seed)
    measured_idx = list(rng.choice(len(pool_params), size=n_init, replace=False))
    best_curve = [max(pool_titers[i] for i in measured_idx)]
    batch_means: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(Path(tmp))
        for _ in range(rounds):
            unmeasured = [i for i in range(len(pool_params)) if i not in set(measured_idx)]
            if strategy == "bo":
                measured = [(pool_params[i], pool_titers[i]) for i in measured_idx]
                cand = [(i, pool_params[i]) for i in unmeasured]
                picked = _propose_from_pool(cfg, measured, cand, q, beta)
            else:  # random baseline
                picked = list(rng.choice(unmeasured, size=q, replace=False))
            batch_means.append(float(np.mean([pool_titers[i] for i in picked])))
            measured_idx += picked
            best_curve.append(max(pool_titers[i] for i in measured_idx))
    return best_curve, batch_means


SEEDS = (7, 11, 23)   # report across seeds; a single seed is not a result


def main() -> None:
    _, opt_titer = sim.true_optimum(n_search=1500)
    # A large fixed library both strategies draw from; the best is a needle, so a small
    # random seed is unlikely to contain it - the optimizer has to work for it.
    pool_params = sim.sample_doe(400, seed=42)
    pool_titers = [sim.endpoint_titer(p) for p in pool_params]
    pool_max = max(pool_titers)
    print(f"True optimum: {opt_titer:.0f} mg/L | best in the 400-candidate library: {pool_max:.0f} mg/L\n")

    base = dict(pool_titers=pool_titers, pool_params=pool_params, rounds=6, n_init=6, q=4, beta=1.5)
    ratios: list[float] = []
    first = None
    for seed in SEEDS:
        bo, bo_batch = run("bo", seed=seed, **base)
        rand, rand_batch = run("random", seed=seed, **base)
        ratios.append(float(np.mean(bo_batch) / np.mean(rand_batch)))
        if first is None:
            first = (bo, rand, bo_batch, rand_batch)

    # One campaign shown in full (illustrative), then the cross-seed robust signal.
    bo, rand, bo_batch, rand_batch = first
    print(f"Example campaign (seed {SEEDS[0]}) - best-found (% of library best) "
          "and mean titer of the batch selected each round:")
    print(f"{'round':>6} {'measured':>9} {'BO best%':>9} {'rand best%':>11} {'BO batch':>10} {'rand batch':>11}")
    for r in range(len(bo)):
        n = base["n_init"] + r * base["q"]
        bb = f"{bo_batch[r-1]:.0f}" if r > 0 else "-"
        rb = f"{rand_batch[r-1]:.0f}" if r > 0 else "-"
        print(f"{r:>6} {n:>9} {100*bo[r]/pool_max:>8.1f}% {100*rand[r]/pool_max:>10.1f}% {bb:>10} {rb:>11}")

    print(f"\nRobust signal across seeds {SEEDS}: BO's proposed batches averaged "
          f"{np.mean(ratios):.2f}x random's titer (range {min(ratios):.2f}-{max(ratios):.2f}x). "
          "Best-found is a weaker discriminator here - a lucky random draw often saturates "
          "it - so batch quality is the honest signal of the acquisition's value.")


if __name__ == "__main__":
    main()

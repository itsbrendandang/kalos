"""Run the closed-loop benchmark sweep and print an honest verdict.

    python -m kalos.bench            # default sweep (2 surfaces x 2 noise levels)
    python -m kalos.bench --quick    # fewer seeds/budget for a fast check
    python -m kalos.bench --pending  # how much of a round a blind re-proposal wastes

Reproducible: every trial is seeded, so the numbers are stable run to run.
"""
from __future__ import annotations

import sys

from .benchmark import run_benchmark, summarize
from .objectives import ackley, gaussian_bump
from .pending import run_pending_duplication


def _fmt(v: float) -> str:
    return f"{v:+.3f}" if v < 0 else f"{v:.3f}"


def _report_pending(quick: bool) -> int:
    """How many recipes of a round a blind mid-round re-proposal repeats."""
    seeds = range(4) if quick else range(10)
    r = run_pending_duplication(seeds=seeds)
    print(f"in-flight recipes re-proposed  (d={r['d']}, q={r['q']}, "
          f"seeds={r['n_seeds']}, duplicate radius={r['duplicate_radius']})")
    print("Question: does a re-proposal mid-round repeat experiments already running?\n")
    print(f"  {'re-proposal':<16} {'per round':>10} {'of q':>6}   per-seed")
    for label, key in (("blind", "blind"), ("pending-aware", "pending_aware")):
        counts = r[key]
        mean = r[f"{key}_mean"]
        print(f"  {label:<16} {mean:>10.2f} {r['q']:>6}   {counts}")
    print(f"\n  -> a blind re-proposal wastes {r['blind_mean']:.2f} of {r['q']} "
          f"recipes per round (worst seed: {r['blind_worst']}); "
          f"pending-aware wastes {r['pending_aware_mean']:.2f}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    if "--pending" in argv:
        return _report_pending(quick)
    seeds = range(4) if quick else range(10)
    budget = 10 if quick else 18
    n_init = 5

    objectives = [gaussian_bump(4), ackley(4)]
    noises = [0.0, 0.15]  # noiseless, then 15%-of-scale measurement noise

    print(f"kalos closed-loop benchmark  (n_init={n_init}, budget={budget}, "
          f"seeds={len(list(seeds))})")
    print("Question: does BO reach a good recipe in fewer experiments than "
          "Latin Hypercube / random?\n")

    lines_for_verdict = []
    for obj in objectives:
        for noise in noises:
            res = run_benchmark(
                obj, budget=budget, n_init=n_init, noise=noise, seeds=seeds
            )
            s = summarize(res, obj, n_init=n_init)
            tag = f"{obj.name}  noise={int(noise * 100)}%  (optimum={obj.optimum:g})"
            print(tag)
            print(f"  {'strategy':<8} {'final best':>12} {'simple regret':>14}")
            for strat in ("bo", "lhs", "random"):
                p = s["per_strategy"][strat]
                print(f"  {strat:<8} {p['final_best_mean']:>12.3f} "
                      f"{p['final_simple_regret']:>14.3f}")
            saved = s["bo_fewer_experiments_than_lhs"]
            if saved is None:
                verdict = "BO did NOT reach the LHS end-value within budget"
            elif saved > 0:
                verdict = f"BO reached LHS's final value ~{saved} experiments sooner"
            else:
                verdict = "BO ~tied LHS (no experiment saving)"
            print(f"  -> {verdict}\n")
            lines_for_verdict.append((obj.name, noise, s, saved))

    # Honest one-paragraph read.
    print("Honest read:")
    for name, noise, s, saved in lines_for_verdict:
        bo = s["per_strategy"]["bo"]["final_simple_regret"]
        lhs = s["per_strategy"]["lhs"]["final_simple_regret"]
        edge = lhs - bo  # positive: BO's regret is lower (better) than LHS
        print(f"  {name} @ {int(noise*100)}% noise: BO regret {bo:.3f} vs LHS "
              f"{lhs:.3f}  (BO better by {edge:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI for the Singleton runner (`docs/M2_INTEGRATION.md`):

  python -m kalos.runner --once            # run every READY experiment, then exit
  python -m kalos.runner --watch 30        # run every READY experiment every 30s
  python -m kalos.runner --id exp_x        # run one experiment now
  python -m kalos.runner --id exp_x --force  # ...and re-run it even if DONE

The backend is selected via `KALOS_BACKEND` (`local` default, or `http`) per
`get_adapter()`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from typing import Sequence

from kalos.runner.adapter import get_adapter
from kalos.runner.singleton import RunResult, run_one, run_ready

log = logging.getLogger("kalos.runner")


def _print_results(results: Sequence[RunResult]) -> None:
    print(json.dumps([dataclasses.asdict(r) for r in results]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kalos.runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run every READY experiment, then exit")
    mode.add_argument("--watch", type=float, metavar="N", help="run every READY experiment every N seconds")
    mode.add_argument("--id", metavar="EXP_ID", help="run one experiment now")
    parser.add_argument(
        "--force", action="store_true",
        help="with --id, re-run even if the experiment is already DONE (discards its prior result)",
    )
    args = parser.parse_args(argv)

    adapter = get_adapter()

    if args.id is not None:
        result = run_one(adapter, args.id, force=args.force)
        _print_results([result])
        return 1 if result.status == "FAILED" else 0

    if args.once:
        results = run_ready(adapter)
        _print_results(results)
        return 1 if any(r.status == "FAILED" for r in results) else 0

    # --watch N
    try:
        while True:
            # A batch failure (e.g. the backend is briefly unreachable) must
            # not kill the daemon - log it and keep polling on the next tick,
            # rather than letting the exception escape the loop.
            try:
                results = run_ready(adapter)
                _print_results(results)
            except Exception:  # noqa: BLE001 - the watch loop must survive a bad poll
                log.exception("run_ready failed during --watch poll; will retry")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

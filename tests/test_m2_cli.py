"""The `python -m kalos.runner` CLI, exercised in-process (no subprocess, so
we do not pay a second cold fastapi/torch import per test). Redirects both the
store's default DB path and the Singleton's default lock path into `tmp_path`
so the real `~/.kalos` is never touched."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

import kalos.runner.__main__ as main_module  # noqa: E402
import kalos.runner.singleton as singleton_module  # noqa: E402
import kalos.store.sqlite_store as sqlite_store_module  # noqa: E402
from kalos.runner.__main__ import main  # noqa: E402
from kalos.store import SqliteStore, Status  # noqa: E402


@pytest.fixture(autouse=True)
def _redirect_kalos_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_store_module, "DEFAULT_DB_PATH", tmp_path / "experiments.db")
    monkeypatch.setattr(singleton_module, "DEFAULT_LOCK_PATH", tmp_path / "runner.lock")
    monkeypatch.delenv("KALOS_BACKEND", raising=False)
    monkeypatch.delenv("KALOS_BACKEND_URL", raising=False)
    return tmp_path


def _good_sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "lipase_titer": titer.round(3),
    })


def _payload_from_df(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}


def test_cli_once_processes_ready_experiments(tmp_path, capsys):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload_from_df(_good_sheet()), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)

    exit_code = main(["--once"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 1 and out[0]["id"] == exp.id and out[0]["status"] == "DONE"
    assert store.get(exp.id).status == Status.DONE


def test_cli_id_runs_one_experiment(tmp_path, capsys):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload_from_df(_good_sheet()), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)

    exit_code = main(["--id", exp.id])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["id"] == exp.id and out[0]["status"] == "DONE"


def test_cli_id_force_flag_reruns_done_experiment(tmp_path, capsys):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload_from_df(_good_sheet()), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    main(["--id", exp.id])
    capsys.readouterr()  # drain first run's output

    exit_code = main(["--id", exp.id, "--force"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["status"] == "DONE"


def test_cli_once_with_nothing_ready_prints_empty_list(tmp_path, capsys):
    exit_code = main(["--once"])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == []


def test_cli_requires_a_mode():
    with pytest.raises(SystemExit):
        main([])


# --- FIX 2(c) (BLOCKER): the --watch loop must survive a bad poll ----------- #
# Before this fix, `run_ready(adapter)` inside the `--watch` loop was not
# wrapped in its own try/except, so ANY exception it raised (a backend outage,
# or - pre FIX 2(a) - even the fallback FAILED-status write inside a single
# experiment's run) propagated straight out of the loop. The outer
# `except KeyboardInterrupt` does not catch a plain exception, so the daemon
# would crash outright instead of logging and continuing to the next poll.

def test_cli_watch_survives_a_run_ready_failure_and_keeps_polling(tmp_path, monkeypatch, capsys):
    calls = {"n": 0}

    def fake_run_ready(adapter):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated backend outage")
        if calls["n"] == 2:
            return []
        raise KeyboardInterrupt  # break the loop on the third tick

    monkeypatch.setattr(main_module, "run_ready", fake_run_ready)
    monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)

    exit_code = main_module.main(["--watch", "0.01"])

    assert exit_code == 0
    assert calls["n"] == 3  # 1st raised (survived), 2nd succeeded, 3rd broke the loop
    # the 2nd (successful) poll's result was still printed
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == []

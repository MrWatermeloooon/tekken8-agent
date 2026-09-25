from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cli():
    spec = importlib.util.spec_from_file_location("training_cli", REPO_ROOT / "scripts" / "training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supervise_records_the_session_and_exit_code(tmp_path):
    cli = _cli()
    run = tmp_path / "run"
    code = cli.supervise([sys.executable, "-c", "print('update=1/1'); import sys; sys.exit(0)"], run, {"seed": 1})
    session = json.loads((run / "session.json").read_text(encoding="utf-8"))
    assert code == 0 and session["status"] == "completed" and session["exit_code"] == 0
    assert session["sleep_prevention"] == "released"
    assert "update=1/1" in (run / "trainer.stdout.log").read_text(encoding="utf-8")

    code = cli.supervise([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], run, {})
    session = json.loads((run / "session.json").read_text(encoding="utf-8"))
    assert code == 3 and session["status"] == "failed"
    assert list(run.glob("session.previous_*.json")), "the earlier session is kept"
    assert "boom" in (run / "trainer.stderr.log").read_text(encoding="utf-8")


def test_keep_awake_is_released_on_every_platform():
    cli = _cli()
    with cli.keep_awake("test") as mechanism:
        assert mechanism in {"system", "systemd-inhibit", "caffeinate", "unavailable"}


def test_process_names_and_stop_refuses_other_processes(tmp_path, monkeypatch):
    cli = _cli()
    assert cli.process_name(os.getpid()) is not None
    assert Path(cli.process_name(os.getpid())).stem.lower().startswith("python")
    run = tmp_path / "run"
    run.mkdir()
    (run / "session.json").write_text(json.dumps({"trainer_pid": os.getpid(), "supervisor_pid": 0}), encoding="utf-8")
    monkeypatch.setattr(cli, "run_path_of", lambda _: run)
    with pytest.raises(SystemExit, match="Refusing to stop"):
        cli.main(["stop", "--run-dir", "ignored"])
    (run / "session.json").write_text(json.dumps({"trainer_pid": 0}), encoding="utf-8")
    assert cli.main(["stop", "--run-dir", "ignored"]) == 0  # not running: nothing to do


def test_newest_build_is_chosen(tmp_path, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, "REPO", tmp_path)
    old = tmp_path / "build" / "Release" / f"t8_v2_train{cli.EXE}"
    new = tmp_path / "build-gpu" / f"t8_v2_train{cli.EXE}"
    for path in (old, new):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    assert cli.find_executable("t8_v2_train") == new
    with pytest.raises(SystemExit, match="not built"):
        cli.find_executable("missing_tool")


def test_run_refuses_to_overwrite_and_validates_shape(tmp_path, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, "find_executable", lambda *_: Path(sys.executable))
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "update_1.t8ppo").write_bytes(b"")
    monkeypatch.setattr(cli, "run_path_of", lambda _: run)
    with pytest.raises(SystemExit, match="already contains"):
        cli.main(["run", "--run-dir", "x"])
    with pytest.raises(SystemExit, match="multiple of 16"):
        cli.main(["run", "--run-dir", "x", "--envs", "100"])
    with pytest.raises(SystemExit, match="minibatch cannot exceed"):
        cli.main(["run", "--run-dir", "x", "--envs", "16", "--horizon", "2", "--minibatch", "64"])


def test_trainer_arguments_match_the_documented_run():
    cli = _cli()
    options = cli.build_parser().parse_args(
        ["run", "--run-dir", "runs/x", "--envs", "32768", "--minibatch", "131072", "--updates", "12000",
         "--extra", "--regression-guard", "pause"])
    arguments = cli.trainer_arguments(options)
    assert arguments[arguments.index("--anneal-updates") + 1] == "12000", "annealing defaults to the run length"
    assert arguments[-2:] == ["--regression-guard", "pause"]
    assert "--seed" not in cli.trainer_arguments(options, include_seed=False)
    assert cli.latest_checkpoint(Path("does-not-exist")) is None

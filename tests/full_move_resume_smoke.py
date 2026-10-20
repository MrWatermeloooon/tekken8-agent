"""Exact-resume and gating smoke test for the full-move trainer (all platforms).

A run resumed from a checkpoint must produce a byte-identical later checkpoint
to an uninterrupted run, through scripted and mirror self-play updates; a
resume with different run-defining options and a run without stand-in geometry
(no character has passed its gate) must be refused.

    python tests/full_move_resume_smoke.py <path to t8_v2_full_move_train>
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    trainer = sys.argv[1]
    root = Path(tempfile.mkdtemp(prefix="t8_full_move_resume_"))

    def run(arguments: list[str]) -> int:
        return subprocess.run([trainer, *arguments], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode

    def train(arguments: list[str]) -> None:
        code = run(arguments)
        if code != 0:
            raise RuntimeError(f"trainer exited with code {code}: {arguments}")

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        common = ["--envs", "64", "--horizon", "8", "--epochs", "1", "--minibatch", "256", "--hidden", "32",
                  "--eval-interval", "2", "--eval-episodes", "16", "--checkpoint-interval", "2",
                  "--anneal-updates", "6", "--learner-character", "jun,king", "--opponents", "jun,king,alisa",
                  "--allow-provisional", "--entropy-target", "6", "--scripted-floor", "0.1", "--draw-penalty", "0.5", "--throw-break-penalty", "0.03", "--time-penalty", "0.01", "--learner-weakness-weight", "0.5", "--scripted-tiers", "--tier-min-updates", "1", "--pool-anneal", "--hold-frames", "6", "--damage-dealt-weight", "0.8", "--damage-taken-weight", "1.2", "--distance-penalty", "0.002", "--champion-interval-steps", "1024",
                  "--champion-rounds", "16",
                  "--self-play-share", "0.5", "--self-play-start", "3", "--seed", "4401"]
        strict = [argument for argument in common if argument != "--allow-provisional"]
        if run(strict + ["--updates", "1", "--run-dir", str(root / "strict")]) == 0:
            raise RuntimeError("the trainer ran without stand-in geometry although no character passed its gate")
        reference, resumed = root / "reference", root / "resumed"
        train(common + ["--updates", "6", "--run-dir", str(reference)])
        train(common + ["--updates", "2", "--run-dir", str(resumed)])
        checkpoint = resumed / "checkpoints" / "update_2.t8ppo"
        if run(common + ["--updates", "6", "--gamma", "0.9", "--run-dir", str(resumed),
                         "--resume", str(checkpoint)]) == 0:
            raise RuntimeError("the trainer resumed with a different gamma")
        train(common + ["--updates", "6", "--run-dir", str(resumed), "--resume", str(checkpoint)])
        for update in (4, 6):
            name = f"update_{update}.t8ppo"
            if sha256(reference / "checkpoints" / name) != sha256(resumed / "checkpoints" / name):
                raise RuntimeError(f"resumed {name} differs from the uninterrupted run")
        modes = (reference / "metrics.jsonl").read_text(encoding="utf-8")
        if "self_play_mirror" not in modes or "scripted_full_move" not in modes:
            raise RuntimeError("the smoke run did not exercise both scripted and self-play updates")
        if not (reference / "champion_history.csv").exists():
            raise RuntimeError("no champion match ran, so the champion state was not tested")
        boosts = [float(line.split('"entropy_boost":')[1].split(",")[0]) for line in modes.splitlines()]
        if max(boosts) <= 1.0:
            raise RuntimeError("the entropy floor never engaged, so its resumed state was not tested")
        # Save and quit at an update between periodic checkpoints (T8Train's Q), then resume.
        quit_run = root / "quit"
        train(common + ["--updates", "6", "--stop-at-update", "3", "--run-dir", str(quit_run)])
        stopped = quit_run / "checkpoints" / "update_3.t8ppo"
        if not stopped.exists() or (quit_run / "checkpoints" / "update_4.t8ppo").exists():
            raise RuntimeError("save and quit did not checkpoint update 3 and stop")
        train(common + ["--updates", "6", "--run-dir", str(quit_run), "--resume", str(stopped)])
        for update in (4, 6):
            name = f"update_{update}.t8ppo"
            if sha256(reference / "checkpoints" / name) != sha256(quit_run / "checkpoints" / name):
                raise RuntimeError(f"{name} after a save-and-quit differs from the uninterrupted run")
        # Resuming a run that went on past its checkpoint (stopped without saving) drops the
        # later log rows instead of duplicating them, and still reproduces the run.
        rerun = root / "rerun"
        shutil.copytree(reference, rerun)
        for later in (rerun / "checkpoints").glob("update_6.*"):  # stopped before update 6 was saved
            later.unlink()
        def logged_updates(run: Path) -> dict[str, list[str]]:
            # Row update numbers (the metrics also hold wall-clock time, so not whole lines).
            metrics = [row.split('"update":')[1].split(",")[0]
                       for row in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            tables = {name: [row.split(",")[0] for row in (run / name).read_text(encoding="utf-8").splitlines()[1:]]
                      for name in ("evaluations.csv", "champion_history.csv")}
            return {"metrics.jsonl": metrics, **tables}
        expected = logged_updates(reference)
        train(common + ["--updates", "6", "--run-dir", str(rerun), "--resume",
                        str(rerun / "checkpoints" / "update_4.t8ppo")])
        for name, updates in logged_updates(rerun).items():
            if updates != expected[name]:
                raise RuntimeError(f"{name} rows after a resume from update 4: {updates}, expected {expected[name]}")
        if sha256(rerun / "checkpoints" / "update_6.t8ppo") != sha256(reference / "checkpoints" / "update_6.t8ppo"):
            raise RuntimeError("update 6 after a resume from update 4 differs from the uninterrupted run")
        # A deeper network with separate policy/value heads resumes exactly too.
        deep = common + ["--shared-layers", "3", "--head-layers", "1"]
        deep_reference, deep_resumed = root / "deep_reference", root / "deep_resumed"
        train(deep + ["--updates", "4", "--run-dir", str(deep_reference)])
        train(deep + ["--updates", "2", "--run-dir", str(deep_resumed)])
        train(deep + ["--updates", "4", "--run-dir", str(deep_resumed), "--resume",
                      str(deep_resumed / "checkpoints" / "update_2.t8ppo")])
        if sha256(deep_reference / "checkpoints" / "update_4.t8ppo") != sha256(deep_resumed / "checkpoints" / "update_4.t8ppo"):
            raise RuntimeError("a deeper network's resumed update 4 differs from the uninterrupted run")
        # A champion rollback restores the champion's entropy boost (and return normalizer) with
        # its weights: thresholds of 1.0 roll back after every match.
        rollback = root / "rollback"
        train(["--envs", "64", "--horizon", "8", "--epochs", "1", "--minibatch", "256", "--hidden", "32",
               "--eval-interval", "2", "--eval-episodes", "16", "--checkpoint-interval", "2",
               "--learner-character", "jun,king", "--opponents", "jun,king", "--allow-provisional",
               "--entropy-target", "6", "--champion-interval-steps", "1024", "--champion-rounds", "16",
               "--champion-promote", "1.0", "--champion-rollback", "1.0", "--champion-rollback-after", "1",
               "--seed", "7", "--updates", "8", "--run-dir", str(rollback)])
        history = (rollback / "champion_history.csv").read_text(encoding="utf-8")
        if history.count("ROLLED BACK") < 2:
            raise RuntimeError("the forced champion rollbacks did not happen")
        boost = {int(row.split('"update":')[1].split(",")[0]): row.split('"entropy_boost":')[1].split(",")[0]
                 for row in (rollback / "metrics.jsonl").read_text(encoding="utf-8").splitlines()}
        if not boost[4] == boost[6] == boost[8] == boost[2]:
            raise RuntimeError(f"rollbacks did not restore the champion's entropy boost: {boost}")
        print("full-move exact resume: byte-identical checkpoints through scripted and self-play updates, "
              "including after a save-and-quit; resumed logs match")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

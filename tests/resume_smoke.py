"""Exact-resume smoke test for the native trainer (all platforms).

Resumed runs must produce byte-identical checkpoints to uninterrupted runs,
including through self-play and a regression-guard rollback; corrupt
metrics and non-finite options must be rejected.

    python tests/resume_smoke.py <path to t8_v2_train>
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    trainer = sys.argv[1]
    root = Path(tempfile.mkdtemp(prefix="t8_v2_resume_"))

    def run(arguments: list[str]) -> int:
        return subprocess.run([trainer, *arguments], stdout=subprocess.DEVNULL).returncode

    def train(arguments: list[str]) -> None:
        code = run(arguments)
        if code != 0:
            raise RuntimeError(f"trainer exited with code {code}: {arguments}")

    def must_fail(arguments: list[str], message: str) -> None:
        if run(arguments) == 0:
            raise RuntimeError(message)

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        common = ["--envs", "32", "--horizon", "8", "--epochs", "1", "--minibatch", "128",
                  "--eval-interval", "1", "--eval-episodes", "16", "--checkpoint-interval", "1",
                  "--reward", "sparse", "--seed", "6501"]
        resumed, reference, corrupt = root / "resumed", root / "reference", root / "corrupt"
        train(common + ["--updates", "1", "--run-dir", str(resumed)])
        checkpoint = resumed / "checkpoints" / "update_1.t8ppo"

        shutil.copytree(resumed, corrupt)
        with (corrupt / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"update":2')
        must_fail(common + ["--updates", "2", "--run-dir", str(corrupt), "--resume",
                            str(corrupt / "checkpoints" / "update_1.t8ppo")],
                  "trainer accepted an incomplete metrics row")
        must_fail(common + ["--updates", "1", "--learning-rate", "NaN", "--run-dir", str(root / "invalid_nan")],
                  "trainer accepted a non-finite learning rate")

        train(common + ["--updates", "2", "--run-dir", str(resumed), "--resume", str(checkpoint)])
        train(common + ["--updates", "2", "--run-dir", str(reference)])
        if sha256(resumed / "checkpoints" / "update_2.t8ppo") != sha256(reference / "checkpoints" / "update_2.t8ppo"):
            raise RuntimeError("resumed checkpoint does not match the uninterrupted checkpoint")

        self_play = ["--envs", "32", "--horizon", "8", "--epochs", "1", "--minibatch", "128",
                     "--eval-interval", "1", "--eval-episodes", "16", "--checkpoint-interval", "1",
                     "--reward", "sparse", "--seed", "7619", "--curriculum-updates", "4"]
        sp_resumed, sp_reference = root / "selfplay_resumed", root / "selfplay_reference"
        train(self_play + ["--updates", "4", "--run-dir", str(sp_resumed)])
        train(self_play + ["--updates", "5", "--run-dir", str(sp_resumed), "--resume",
                           str(sp_resumed / "checkpoints" / "update_4.t8ppo")])
        train(self_play + ["--updates", "5", "--run-dir", str(sp_reference)])
        if sha256(sp_resumed / "checkpoints" / "update_5.t8ppo") != sha256(sp_reference / "checkpoints" / "update_5.t8ppo"):
            raise RuntimeError("resumed self-play checkpoint does not match the uninterrupted checkpoint")

        # Self-play stability options: scripted league blocks, an older-checkpoint pool, and the
        # entropy floor (whose adaptive multiplier lives in the trainer state) resume exactly.
        stable = ["--envs", "32", "--horizon", "8", "--epochs", "1", "--minibatch", "128",
                  "--eval-interval", "1", "--eval-episodes", "16", "--checkpoint-interval", "1",
                  "--reward", "sparse", "--seed", "8111", "--curriculum-updates", "4",
                  "--league-scripted-share", "0.5", "--league-block-updates", "2", "--league-pool-size", "3",
                  "--entropy-target", "3.5"]  # above a fresh network's per-choice entropy (ln 24)
        st_resumed, st_reference = root / "stable_resumed", root / "stable_reference"
        train(stable + ["--updates", "7", "--run-dir", str(st_resumed)])
        train(stable + ["--updates", "9", "--run-dir", str(st_resumed), "--resume",
                        str(st_resumed / "checkpoints" / "update_7.t8ppo")])
        train(stable + ["--updates", "9", "--run-dir", str(st_reference)])
        if sha256(st_resumed / "checkpoints" / "update_9.t8ppo") != sha256(st_reference / "checkpoints" / "update_9.t8ppo"):
            raise RuntimeError("resumed run with the stability options does not match the uninterrupted run")
        must_fail(stable[:-2] + ["--updates", "10", "--run-dir", str(st_resumed), "--resume",
                                 str(st_resumed / "checkpoints" / "update_9.t8ppo")],
                  "trainer resumed with different stability options")
        rows = [row for row in (st_reference / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if '"entropy_coefficient"' in row]
        coefficients = [float(re.search(r'"entropy_coefficient":([0-9.eE+-]+)', row).group(1)) for row in rows]
        if not coefficients or coefficients[-1] <= coefficients[0]:
            raise RuntimeError(f"the entropy floor did not raise the coefficient: {coefficients}")
        opponents = [re.search(r'"training_opponent":"([a-z_0-9]+)"', row).group(1) for row in rows]
        if "scripted" not in opponents[4:] or not any(name.startswith("self_play") for name in opponents):
            raise RuntimeError(f"league updates did not mix scripted and self-play opponents: {opponents}")

        # Regression guard: thresholds that trigger on any held-out drop. The reference run
        # rolls back and then pauses (exit 3); a run stopped exactly at the rollback must resume
        # into the same checkpoint.
        guard = ["--envs", "64", "--horizon", "16", "--epochs", "1", "--minibatch", "256",
                 "--eval-interval", "1", "--eval-episodes", "32", "--checkpoint-interval", "1",
                 "--seed", "9123", "--regression-score-drop", "0", "--regression-style-drop", "1",
                 "--regression-max-side-gap", "1", "--regression-patience", "1",
                 "--regression-max-rollbacks", "1"]
        guard_reference, guard_resumed = root / "guard_reference", root / "guard_resumed"
        code = run(guard + ["--updates", "30", "--run-dir", str(guard_reference)])
        if code != 3:
            raise RuntimeError(f"regression guard did not pause with exit code 3 (got {code})")
        rows = (guard_reference / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        rollbacks = [row for row in rows if '"action":"rollback"' in row]
        if not rollbacks:
            raise RuntimeError("regression guard never rolled back before pausing")
        if '"action":"pause"' not in rows[-1]:
            raise RuntimeError("final guard row is not a pause")
        pause_update = int(re.search(r'"update":(\d+)', rows[-1]).group(1))
        rollback_update = int(re.search(r'"update":(\d+)', rollbacks[0]).group(1))
        train(guard + ["--updates", str(rollback_update), "--run-dir", str(guard_resumed)])
        code = run(guard + ["--updates", "30", "--run-dir", str(guard_resumed), "--resume",
                            str(guard_resumed / "checkpoints" / f"update_{rollback_update}.t8ppo")])
        if code != 3:
            raise RuntimeError(f"resumed guard run did not pause (exit {code})")
        if (sha256(guard_reference / "checkpoints" / f"update_{pause_update}.t8ppo") !=
                sha256(guard_resumed / "checkpoints" / f"update_{pause_update}.t8ppo")):
            raise RuntimeError("checkpoint after a resumed rollback does not match the uninterrupted run")
    except RuntimeError as error:
        print(f"resume smoke failed: {error}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("resume smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

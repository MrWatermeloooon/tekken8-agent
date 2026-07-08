from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> int:
    print("+ " + " ".join(command))
    result = subprocess.run(command, check=False)
    return result.returncode


def main() -> int:
    checks = [
        [sys.executable, "-m", "compileall", "src", "tests", "scripts"],
        [sys.executable, "-m", "pytest", "-q"],
        [
            sys.executable,
            "scripts/visualize_sim.py",
            "--p1",
            "scripted",
            "--p1-scripted",
            "rushdown",
            "--p2",
            "scripted",
            "--p2-scripted",
            "rushdown",
            "--headless-steps",
            "120",
        ],
        [
            sys.executable,
            "scripts/run_sim_episode.py",
            "--episodes",
            "3",
        ],
    ]
    for command in checks:
        code = run(command)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

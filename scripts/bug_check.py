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
            "--headless-steps",
            "120",
            "--checkpoint",
            "checkpoints/sim_linear_policy.npz",
        ],
        [
            sys.executable,
            "scripts/evaluate_sim_policy.py",
            "--checkpoint",
            "checkpoints/sim_linear_policy.npz",
            "--episodes",
            "10",
            "--max-decisions",
            "1000",
        ],
    ]
    for command in checks:
        code = run(command)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

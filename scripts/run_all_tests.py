"""Every test that does not need Tekken 8, one after another.

    run_all_tests.bat                 (double-click; wraps this script)
    python scripts/run_all_tests.py [--skip-build] [--quick]

Steps (each logs to test_logs/<step>.log; a failure never stops the rest):

  1  Build (CUDA)                  every native target
  2  Native tests (CUDA)           ctest: engine refactor, CPU/GPU parity, PPO math, feature
                                   tables, normalization, full-roster smoke, exact resume, ...
  3  Python tests                  pytest: catalog, parser, provenance, live runtime, tools
  4  CPU-only build and tests      the CPU targets without CUDA
  5  Generated data reproducible   catalog, bindings, candidates, stance skeletons
  6  Compute Sanitizer             memcheck, initcheck, racecheck on the full-move simulator
  7  Training smoke runs           full-move (every character) and compatibility trainers
  8  Acceptance report             informational: what still needs the game and long runs
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "test_logs"
EXE = ".exe" if os.name == "nt" else ""


DATA_SNAPSHOT = r"""
import hashlib, json, sys
from pathlib import Path
files = {str(p): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
         for root in ("data/generated", "data/character_modules") for p in sorted(Path(root).rglob("*")) if p.is_file()}
if sys.argv[2] == "save":
    Path(sys.argv[1]).write_text(json.dumps(files))
    sys.exit(0)
before = json.loads(Path(sys.argv[1]).read_text())
changed = sorted(set(before) ^ set(files) | {p for p in files if p in before and before[p] != files[p]})
for path in changed:
    print("changed by regeneration:", path)
print(f"{len(files)} generated files, {len(changed)} changed")
sys.exit(1 if changed else 0)
"""


def binary(name: str) -> str:
    for directory in ("build-gpu/Release", "build-gpu", "build/Release", "build"):
        path = REPO / directory / f"{name}{EXE}"
        if path.exists():
            return str(path)
    return str(REPO / "build-gpu" / "Release" / f"{name}{EXE}")


def find_sanitizer() -> str | None:
    found = shutil.which("compute-sanitizer")
    if found:
        return found
    pattern = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA GPU Computing Toolkit",
                           "CUDA", "*", "compute-sanitizer", "compute-sanitizer.exe")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def run_step(name: str, title: str, commands: list[list[str]]) -> tuple[str, float]:
    print(f"\n---- {title} ----", flush=True)
    started = time.perf_counter()
    status = "PASS"
    with (LOGS / f"{name}.log").open("w", encoding="utf-8", errors="replace") as log:
        for command in commands:
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                status = "FAIL"
                log.write(f"\n(exit code {result.returncode})\n")
                break
    seconds = time.perf_counter() - started
    print(f"[{status}] {title}  ({seconds:.0f} s{', see test_logs/' + name + '.log' if status == 'FAIL' else ''})",
          flush=True)
    return status, seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--quick", action="store_true", help="skip the CPU-only build and Compute Sanitizer")
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    python = sys.executable
    scratch = Path(tempfile.mkdtemp(prefix="t8_tests_"))
    results: list[tuple[str, str, float]] = []
    print("========================================")
    print(" Tekken 8 agent: every test before the game")
    print("========================================")

    def step(name: str, title: str, commands: list[list[str]]) -> None:
        status, seconds = run_step(name, title, commands)
        results.append((status, title, seconds))

    # A stale pytest folder left by an interrupted run blocks the live-policy test.
    shutil.rmtree(REPO / "build-gpu" / "pytest-live-policy", ignore_errors=True)
    if not args.skip_build:
        step("1_build", "Build (CUDA)", [["cmake", "--build", "build-gpu", "--config", "Release", "--parallel"]])
    step("2_native", "Native tests (CUDA)", [["ctest", "--test-dir", "build-gpu", "-C", "Release",
                                               "--output-on-failure"]])
    step("3_python", "Python tests", [[python, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                                       "--basetemp", str(scratch / "pytest")]])
    if not args.quick:
        step("4_cpu", "CPU-only build and tests", [
            ["cmake", "-S", ".", "-B", "build-cpu", "-DT8_V2_ENABLE_CUDA=OFF"],
            ["cmake", "--build", "build-cpu", "--config", "Release", "--parallel"],
            ["ctest", "--test-dir", "build-cpu", "-C", "Release", "--output-on-failure"]])
    # Regenerating must reproduce the files on disk exactly (not the last commit: the data may have
    # uncommitted changes on purpose).
    snapshot = [python, "-c", DATA_SNAPSHOT, str(scratch / "data.json")]
    step("5_data", "Generated data reproducible", [
        snapshot + ["save"],
        [python, "tools/compile_full_move_catalog.py"],
        [python, "tools/export_full_combat_bindings.py"],
        [python, "tools/generate_state_machines.py", "--check"],
        snapshot + ["compare"]])
    sanitizer = None if args.quick else find_sanitizer()
    if sanitizer:
        smoke = [binary("t8_v2_full_roster_smoke"), "data/generated", "10"]
        step("6_sanitizer", "Compute Sanitizer", [
            [sanitizer, "--tool", tool, "--error-exitcode", "1", *smoke] for tool in ("memcheck", "initcheck", "racecheck")
        ] + [[sanitizer, "--tool", "memcheck", "--error-exitcode", "1", binary("t8_v2_full_move_gpu_tests"),
              "data/generated", "600"]])
    elif not args.quick:
        results.append(("SKIP", "Compute Sanitizer (compute-sanitizer not found)", 0.0))
    step("7_training", "Training smoke runs", [
        [binary("t8_v2_full_move_train"), "--smoke", "--allow-provisional", "--learner-character", "all",
         "--run-dir", str(scratch / "full_move")],
        [binary("t8_v2_train"), "--smoke", "--opponents", "roster", "--learner-character", "jun",
         "--observation-mode", "visual", "--reward", "shaped", "--seed", "2027", "--run-dir",
         str(scratch / "compatibility")]])
    with (LOGS / "8_acceptance.log").open("w", encoding="utf-8") as log:
        subprocess.run([python, "tools/acceptance_report.py", "--output", str(LOGS / "acceptance")], cwd=REPO,
                       stdout=log, stderr=subprocess.STDOUT)
    results.append(("INFO", "Acceptance report (test_logs/acceptance.md)", 0.0))
    shutil.rmtree(scratch, ignore_errors=True)

    print("\n========================================")
    print(" Summary")
    print("========================================")
    for status, title, seconds in results:
        print(f"  {status:<4}  {title}" + (f"  ({seconds:.0f} s)" if seconds else ""))
    failed = sum(status == "FAIL" for status, _, _ in results)
    print()
    print("All tests passed. What is left needs the game (Practice mode) or long training runs."
          if failed == 0 else f"{failed} step(s) failed; the logs are in test_logs/.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

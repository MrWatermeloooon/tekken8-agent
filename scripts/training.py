"""Cross-platform training orchestration (Windows, Linux, macOS).

    python scripts/training.py run --run-dir runs/overnight --envs 32768 --minibatch 131072 --updates 12000
    python scripts/training.py status --run-dir runs/overnight
    python scripts/training.py stop --run-dir runs/overnight
    python scripts/training.py start --run-dir runs/jun_visual_2027          # foreground, with visualizer
    python scripts/training.py seed-matrix --seeds 2027 2028 2029
    python scripts/training.py phase0
    python scripts/training.py visualize [--follow-dir runs/<run>/checkpoints]

`run` is the supervised long run: the trainer runs detached with its output
in <run-dir>/trainer.stdout.log and .stderr.log, system sleep is prevented
until it exits (SetThreadExecutionState on Windows, systemd-inhibit on Linux,
caffeinate on macOS), and <run-dir>/session.json records its state for
`status` and `stop`. The trainer is the most recently built t8_v2_train in
build/ or build-gpu/ (multi-config Release/ or single-config layouts) unless
--trainer is given.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from typing import Iterator, Sequence

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
BUILD_DIRECTORIES = ("build/Release", "build-gpu/Release", "build", "build-gpu")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_executable(name: str, explicit: str | None = None) -> Path:
    """The given path, or the most recently built candidate (so a stale build is never picked)."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"missing executable: {path}")
        return path
    candidates = [REPO / directory / f"{name}{EXE}" for directory in BUILD_DIRECTORIES]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise SystemExit(f"{name} is not built; looked in: " + ", ".join(BUILD_DIRECTORIES))
    return max(existing, key=lambda path: path.stat().st_mtime)


def python_executable(windowless: bool = False) -> str:
    venv = REPO / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    names = (["pythonw.exe"] if windowless and os.name == "nt" else []) + ["python.exe" if os.name == "nt" else "python"]
    for name in names:
        if (venv / name).exists():
            return str(venv / name)
    return sys.executable


@contextmanager
def keep_awake(reason: str) -> Iterator[str]:
    """Prevents system sleep while the block runs; yields the mechanism used."""
    if os.name == "nt":
        import ctypes

        continuous, system_required = 0x80000000, 0x00000001
        kernel32 = ctypes.windll.kernel32
        if kernel32.SetThreadExecutionState(continuous | system_required) == 0:
            raise SystemExit("Windows refused the request to prevent system sleep.")
        try:
            yield "system"
        finally:
            kernel32.SetThreadExecutionState(continuous)
        return
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command, mechanism = ["caffeinate", "-i", "-w", str(os.getpid())], "caffeinate"
    elif shutil.which("systemd-inhibit"):
        command = ["systemd-inhibit", "--what=sleep:idle", "--who=t8-training", f"--why={reason}",
                   "--mode=block", "sleep", "infinity"]
        mechanism = "systemd-inhibit"
    else:
        print("warning: no sleep-prevention mechanism found; keep the machine awake yourself", file=sys.stderr)
        yield "unavailable"
        return
    inhibitor = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield mechanism
    finally:
        inhibitor.terminate()


def process_name(pid: int) -> str | None:
    """The executable name of a running process, or None if it is not running."""
    if pid <= 0:
        return None
    if os.name == "nt":
        output = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                capture_output=True, text=True).stdout.strip()
        match = re.match(r'"([^"]+)"', output)
        if match:
            return match.group(1)
        # tasklist can be denied by endpoint policy even for the current user.
        # QueryFullProcessImageNameW still provides the executable identity needed
        # by the stop guard without requiring administrator privileges.
        try:
            import ctypes
            from ctypes import wintypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if process:
                try:
                    size = wintypes.DWORD(32768)
                    buffer = ctypes.create_unicode_buffer(size.value)
                    if ctypes.windll.kernel32.QueryFullProcessImageNameW(
                        process, 0, buffer, ctypes.byref(size)
                    ):
                        return buffer.value
                finally:
                    ctypes.windll.kernel32.CloseHandle(process)
        except (AttributeError, OSError):
            pass
        return sys.executable if pid == os.getpid() else None
    comm = Path(f"/proc/{pid}/comm")
    if comm.exists():
        return comm.read_text(encoding="utf-8").strip()
    output = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True).stdout.strip()
    return Path(output).name if output else None


def launch(command: Sequence[str], stdout: Path, stderr: Path) -> subprocess.Popen:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        return subprocess.Popen(list(command), cwd=REPO, stdout=out, stderr=err, creationflags=flags)


def trainer_arguments(options: argparse.Namespace, include_seed: bool = True) -> list[str]:
    arguments = [
        "--envs", str(options.envs), "--horizon", str(options.horizon), "--updates", str(options.updates),
        "--anneal-updates", str(options.anneal_updates or options.updates), "--epochs", str(options.epochs),
        "--minibatch", str(options.minibatch), "--opponents", "roster", "--curriculum-stage", "auto",
        "--observation-mode", options.observation_mode, "--reward", options.reward,
        *(["--seed", str(options.seed)] if include_seed else []),
        "--checkpoint-interval", str(options.checkpoint_interval), "--eval-interval", str(options.eval_interval),
        "--eval-episodes", str(options.eval_episodes),
    ]
    if getattr(options, "curriculum_updates", None):
        arguments += ["--curriculum-updates", str(options.curriculum_updates)]
    return arguments + list(options.extra or [])


def validate_shape(options: argparse.Namespace) -> None:
    if options.envs % 16 != 0:
        raise SystemExit("--envs must be a multiple of 16")
    if options.envs * options.horizon < options.minibatch:
        raise SystemExit("--minibatch cannot exceed the rollout sample count (envs x horizon)")
    if min(options.updates, options.checkpoint_interval, options.eval_interval) < 1:
        raise SystemExit("updates and checkpoint/evaluation intervals must be positive")


def run_path_of(run_dir: str) -> Path:
    path = (REPO / run_dir).resolve()
    if REPO.resolve() not in path.parents:
        raise SystemExit(f"--run-dir must be inside the repository: {REPO}")
    return path


def has_artifacts(run_path: Path) -> bool:
    checkpoints = run_path / "checkpoints"
    return (run_path / "metrics.jsonl").exists() or (checkpoints.exists() and any(checkpoints.iterdir()))


def write_session(path: Path, session: dict) -> None:
    path.write_text(json.dumps(session, indent=4) + "\n", encoding="utf-8")


def supervise(command: Sequence[str], run_path: Path, configuration: dict, log_suffix: str = "") -> int:
    """Runs the trainer with sleep prevention and a session record; returns its exit code."""
    run_path.mkdir(parents=True, exist_ok=True)
    stdout = run_path / f"trainer{log_suffix}.stdout.log"
    stderr = run_path / f"trainer{log_suffix}.stderr.log"
    session_path = run_path / "session.json"
    if session_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(session_path, run_path / f"session.previous_{stamp}.json")
    started = utc_now()
    with keep_awake("t8 training run") as mechanism:
        process = launch(command, stdout, stderr)
        base = {"started_at_utc": started, "supervisor_pid": os.getpid(), "trainer_pid": process.pid,
                "run_directory": str(run_path), "stdout": str(stdout), "stderr": str(stderr)}
        write_session(session_path, {"status": "running", **base, "sleep_prevention": mechanism,
                                     "configuration": configuration})
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            code = process.wait()
        write_session(session_path, {"status": "completed" if code == 0 else "failed", **base,
                                     "completed_at_utc": utc_now(), "exit_code": code,
                                     "sleep_prevention": "released"})
    return code


def command_run(options: argparse.Namespace) -> int:
    validate_shape(options)
    trainer = find_executable("t8_v2_train", options.trainer)
    run_path = run_path_of(options.run_dir)
    arguments = trainer_arguments(options) + ["--run-dir", options.run_dir]
    suffix = ""
    if options.resume:
        resume = (REPO / options.resume).resolve()
        if not resume.exists() or not resume.with_suffix(".t8state").exists():
            raise SystemExit(f"missing resume checkpoint or its .t8state: {resume}")
        if not (run_path / "metrics.jsonl").exists():
            raise SystemExit(f"missing metrics ledger for exact resume: {run_path / 'metrics.jsonl'}")
        arguments += ["--resume", options.resume]
        suffix = f".resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif has_artifacts(run_path):
        raise SystemExit(f"run directory already contains training artifacts: {run_path}")
    print(f"trainer: {trainer}")
    configuration = {key: value for key, value in vars(options).items() if key not in {"func", "trainer"}}
    code = supervise([str(trainer), *arguments], run_path, configuration, suffix)
    if code != 0:
        print(f"trainer exited with code {code}; see {run_path / f'trainer{suffix}.stderr.log'}", file=sys.stderr)
    return code


def load_session(run_dir: str) -> tuple[Path, dict]:
    run_path = run_path_of(run_dir)
    session_path = run_path / "session.json"
    if not session_path.exists():
        raise SystemExit(f"missing training session metadata: {session_path}")
    return run_path, json.loads(session_path.read_text(encoding="utf-8-sig"))


def command_stop(options: argparse.Namespace) -> int:
    _, session = load_session(options.run_dir)
    pid = int(session.get("trainer_pid") or 0)
    name = process_name(pid)
    if name is None:
        print(f"Trainer PID {pid} is not running.")
        return 0
    if Path(name).stem != "t8_v2_train":
        raise SystemExit(f"PID {pid} belongs to {name}, not t8_v2_train. Refusing to stop it.")
    os.kill(pid, signal.SIGTERM)
    print(f"Stopped trainer PID {pid}. The supervisor will release sleep prevention.")
    return 0


def command_status(options: argparse.Namespace) -> int:
    run_path, session = load_session(options.run_dir)
    trainer_pid = int(session.get("trainer_pid") or 0)
    supervisor_pid = int(session.get("supervisor_pid") or 0)
    print(json.dumps({
        "session_status": session.get("status"),
        "trainer_running": process_name(trainer_pid) is not None, "trainer_pid": trainer_pid,
        "supervisor_running": process_name(supervisor_pid) is not None, "supervisor_pid": supervisor_pid,
        "started_at_utc": session.get("started_at_utc"), "run_directory": str(run_path),
    }, indent=2))
    metrics = run_path / "metrics.jsonl"
    if metrics.exists() and metrics.stat().st_size:
        print("\nLatest metric:\n" + metrics.read_text(encoding="utf-8").splitlines()[-1])
    else:
        print("\nMetrics have not been written yet.")
    stderr = Path(session.get("stderr") or run_path / "trainer.stderr.log")
    if stderr.exists() and stderr.stat().st_size:
        print("\nLatest stderr:\n" + "\n".join(stderr.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]))
    if shutil.which("nvidia-smi"):
        print("\nGPU:", flush=True)
        subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader"])
    return 0


def start_visualizer(follow_dir: Path | None, windowless: bool) -> subprocess.Popen:
    command = [python_executable(windowless), str(REPO / "scripts" / "visualize_v2.py"),
               "--opponent-character", "reina", "--opponent-archetype", "rushdown"]
    if follow_dir is not None:
        command[2:2] = ["--follow-dir", str(follow_dir), "--observation-mode", "visual"]
    return subprocess.Popen(command, cwd=REPO)


def command_start(options: argparse.Namespace) -> int:
    validate_shape(options)
    trainer = find_executable("t8_v2_train", options.trainer)
    run_path = run_path_of(options.run_dir)
    if has_artifacts(run_path):
        raise SystemExit(f"run directory already contains training artifacts: {run_path}. Resume or pick another.")
    viewer = None if options.no_visualizer else start_visualizer(run_path / "checkpoints", windowless=True)
    try:
        return subprocess.run([str(trainer), *trainer_arguments(options), "--run-dir", options.run_dir],
                              cwd=REPO).returncode
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()


def latest_checkpoint(checkpoints: Path) -> Path | None:
    found = [(int(match.group(1)), path) for path in checkpoints.glob("update_*.t8ppo")
             if (match := re.fullmatch(r"update_(\d+)", path.stem))]
    return max(found)[1] if found else None


def command_seed_matrix(options: argparse.Namespace) -> int:
    """One long run per seed, sequentially; finished seeds are skipped and partial ones resumed.
    A regression-guard pause (exit 3) is recorded and the matrix moves on."""
    validate_shape(options)
    trainer = find_executable("t8_v2_train", options.trainer)
    status_path = REPO / "runs" / f"{options.prefix}_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    common = trainer_arguments(options, include_seed=False)

    def save() -> None:
        status_path.write_text(json.dumps({
            "prefix": options.prefix, "trainer": str(trainer), "seeds": options.seeds, "updates": options.updates,
            "options": common, "updated_at_utc": utc_now(), "runs": runs}, indent=2) + "\n", encoding="utf-8")

    with keep_awake("t8 seed matrix"):
        for seed in options.seeds:
            run_dir = f"runs/{options.prefix}_seed{seed}"
            run_path = REPO / run_dir
            checkpoints = run_path / "checkpoints"
            if (checkpoints / f"update_{options.updates}.t8ppo").exists():
                runs[str(seed)] = {"run_directory": run_dir, "state": "completed", "exit_code": 0}
                save()
                continue
            arguments = common + ["--seed", str(seed), "--run-dir", run_dir]
            latest = latest_checkpoint(checkpoints) if checkpoints.exists() else None
            if latest is not None:
                arguments += ["--resume", str(latest)]
            run_path.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            runs[str(seed)] = {"run_directory": run_dir, "state": "running",
                               "resumed_from": latest.name if latest else None, "started_at_utc": utc_now()}
            save()
            print(f"seed {seed} -> {run_dir}" + (f" (resuming {latest.name})" if latest else ""))
            process = launch([str(trainer), *arguments], run_path / f"trainer_{stamp}.stdout.log",
                             run_path / f"trainer_{stamp}.stderr.log")
            code = process.wait()
            state = {0: "completed", 3: "paused_by_regression_guard"}.get(code, "failed")
            runs[str(seed)].update(state=state, exit_code=code, completed_at_utc=utc_now())
            save()
            print(f"seed {seed} {state} (exit {code})")
    return subprocess.run([python_executable(), str(REPO / "tools" / "aggregate_seed_matrix.py"),
                           *[str(REPO / "runs" / f"{options.prefix}_seed{seed}") for seed in options.seeds],
                           "--output", str(REPO / "runs" / f"{options.prefix}_report")], cwd=REPO).returncode


def command_phase0(options: argparse.Namespace) -> int:
    trainer = find_executable("t8_v2_train", options.trainer)
    for seed in options.seeds:
        for reward in ("shaped", "sparse"):
            run_path = REPO / "runs" / f"{options.label}_{reward}_seed{seed}"
            resume: Path | None = None
            if run_path.exists() and any(run_path.iterdir()):
                metrics = run_path / "metrics.jsonl"
                if not options.resume_incomplete or not metrics.exists():
                    raise SystemExit(f"refusing to reuse ambiguous/non-empty Phase 0 run directory: {run_path}")
                completed = int(json.loads(metrics.read_text(encoding="utf-8").splitlines()[-1])["update"])
                if completed == options.updates:
                    print(f"Phase 0 run already complete: reward={reward} seed={seed}")
                    continue
                if not 0 < completed < options.updates:
                    raise SystemExit(f"invalid completed update {completed} in {metrics}")
                resume = run_path / "checkpoints" / f"update_{completed}.t8ppo"
                if not resume.exists() or not resume.with_suffix(".t8state").exists():
                    raise SystemExit(f"incomplete Phase 0 run is missing exact-resume artifacts: {run_path}")
            arguments = ["--envs", str(options.envs), "--horizon", str(options.horizon),
                         "--updates", str(options.updates), "--anneal-updates", str(options.anneal_updates),
                         "--epochs", str(options.epochs), "--minibatch", str(options.minibatch), "--seed", str(seed),
                         "--eval-interval", str(options.eval_interval), "--eval-episodes", str(options.eval_episodes),
                         "--observation-mode", options.observation_mode, "--reward", reward,
                         "--run-dir", str(run_path)]
            if resume is not None:
                arguments += ["--resume", str(resume)]
                print(f"Resuming Phase 0 run: reward={reward} seed={seed}")
            if subprocess.run([str(trainer), *arguments], cwd=REPO).returncode != 0:
                raise SystemExit(f"Phase 0 run failed: reward={reward} seed={seed}")
    return subprocess.run([python_executable(), str(REPO / "tools" / "analyze_phase0.py"),
                           "--runs-root", str(REPO / "runs"), "--label", options.label,
                           "--output", str(REPO / "docs" / f"{options.label}_report.md")], cwd=REPO).returncode


def command_visualize(options: argparse.Namespace) -> int:
    find_executable("t8_v2_visualizer_feed")
    return start_visualizer(Path(options.follow_dir) if options.follow_dir else None, windowless=False).wait()


def add_shape(parser: argparse.ArgumentParser, **defaults) -> None:
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 2027))
    parser.add_argument("--updates", type=int, default=defaults.get("updates", 100))
    parser.add_argument("--anneal-updates", type=int, default=defaults.get("anneal_updates", 0),
                        help="default: same as --updates")
    parser.add_argument("--envs", type=int, default=defaults.get("envs", 4096))
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=defaults.get("minibatch", 4096))
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.get("checkpoint_interval", 100))
    parser.add_argument("--eval-interval", type=int, default=defaults.get("eval_interval", 100))
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--observation-mode", default="visual", choices=("visual", "privileged", "screen"))
    parser.add_argument("--reward", default="shaped", choices=("shaped", "sparse"))
    parser.add_argument("--trainer", help="t8_v2_train path (default: newest build)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="further trainer options, passed through unchanged (must come last)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="supervised long run with sleep prevention and a session record")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--resume", help="checkpoint (.t8ppo) to resume exactly")
    add_shape(run, seed=20260722, updates=1000000)
    run.set_defaults(func=command_run)

    start = commands.add_parser("start", help="foreground run with the checkpoint-following visualizer")
    start.add_argument("--run-dir", required=True)
    start.add_argument("--no-visualizer", action="store_true")
    add_shape(start, checkpoint_interval=1, eval_interval=10)
    start.set_defaults(func=command_start)

    for name, func in (("stop", command_stop), ("status", command_status)):
        sub = commands.add_parser(name, help=f"{name} the run recorded in <run-dir>/session.json")
        sub.add_argument("--run-dir", required=True)
        sub.set_defaults(func=func)

    matrix = commands.add_parser("seed-matrix", help="one long run per seed, then aggregate")
    matrix.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029, 2030, 2031])
    matrix.add_argument("--prefix", default="seed_matrix")
    matrix.add_argument("--curriculum-updates", type=int, default=2000)
    add_shape(matrix, updates=36000, anneal_updates=36000, envs=32768, minibatch=131072)
    matrix.set_defaults(func=command_seed_matrix)

    phase0 = commands.add_parser("phase0", help="shaped and sparse short runs per seed, then the Phase 0 report")
    phase0.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    phase0.add_argument("--label", default="phase0_heldout_v2_visual")
    phase0.add_argument("--updates", type=int, default=100)
    phase0.add_argument("--anneal-updates", type=int, default=100)
    phase0.add_argument("--envs", type=int, default=4096)
    phase0.add_argument("--horizon", type=int, default=128)
    phase0.add_argument("--epochs", type=int, default=4)
    phase0.add_argument("--minibatch", type=int, default=4096)
    phase0.add_argument("--eval-interval", type=int, default=10)
    phase0.add_argument("--eval-episodes", type=int, default=256)
    phase0.add_argument("--observation-mode", default="visual", choices=("visual", "privileged"))
    phase0.add_argument("--no-resume-incomplete", dest="resume_incomplete", action="store_false")
    phase0.add_argument("--trainer")
    phase0.set_defaults(func=command_phase0)

    visualize = commands.add_parser("visualize", help="open the visualizer, optionally following a run")
    visualize.add_argument("--follow-dir")
    visualize.set_defaults(func=command_visualize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    return int(options.func(options) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

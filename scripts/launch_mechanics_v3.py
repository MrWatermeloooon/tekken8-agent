from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000


def vecnormalize_sidecar(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}.vecnormalize.pkl")


def latest_complete_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        return None
    normalized = [path for path in checkpoints if vecnormalize_sidecar(path).exists()]
    return (normalized or checkpoints)[0]


def command_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def launch(command: list[str], *, stdout: Path | None = None, stderr: Path | None = None, hidden: bool) -> int:
    env = os.environ.copy()
    path_value = None
    for key in list(env):
        if key.upper() == "PATH":
            path_value = env[key]
            if key != "PATH":
                del env[key]
    if path_value is not None:
        env["PATH"] = path_value
    env["PYTHONPATH"] = str(ROOT / "src")
    stdout_handle = stdout.open("w", encoding="utf-8") if stdout else subprocess.DEVNULL
    stderr_handle = stderr.open("w", encoding="utf-8") if stderr else subprocess.DEVNULL
    flags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    if hidden:
        flags |= CREATE_NO_WINDOW
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=False,
    )
    return int(process.pid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch long-running Tekken helper processes.")
    parser.add_argument(
        "target",
        choices=[
            "train",
            "train-grabs",
            "train-grabs-fast",
            "train-winloss-fast",
            "train-anti-stall-fast",
            "train-stalemate-fast",
            "train-mixed-fast",
            "visualizer",
            "visualizer-grabs",
            "visualizer-grabs-fast",
            "visualizer-winloss-fast",
            "visualizer-anti-stall-fast",
            "visualizer-stalemate-fast",
            "visualizer-mixed-fast",
            "live-old",
            "live-new",
        ],
    )
    args = parser.parse_args()

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    (ROOT / "runs" / "full_selfplay_mechanics_v3").mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints" / "full_selfplay_mechanics_v3").mkdir(parents=True, exist_ok=True)

    if args.target == "train":
        pid = launch(
            [
                str(PYTHON),
                "scripts/train_sim_ppo_selfplay.py",
                "--iterations",
                "20",
                "--timesteps-per-iteration",
                "32768",
                "--full-self-play",
                "--bootstrap-iterations",
                "1",
                "--elo-sampling",
                "--latest-checkpoint-rate",
                "0.80",
                "--best-checkpoint-rate",
                "0.20",
                "--n-envs",
                "8",
                "--n-steps",
                "256",
                "--batch-size",
                "512",
                "--checkpoint-dir",
                "checkpoints/full_selfplay_mechanics_v3",
                "--final-checkpoint",
                "checkpoints/sim_ppo_selfplay_mechanics_v3.zip",
                "--run-dir",
                "runs/full_selfplay_mechanics_v3",
            ],
            stdout=logs / "full_selfplay_mechanics_v3.out.log",
            stderr=logs / "full_selfplay_mechanics_v3.err.log",
            hidden=True,
        )
        print(f"TRAIN_PID={pid}")
        return 0

    if args.target == "train-grabs":
        fast = False
        full_self_play = True
        run_name = "full_selfplay_grabs_v4"
        checkpoint_name = "full_selfplay_grabs_v4"
        final_name = "sim_ppo_selfplay_grabs_v4.zip"
        log_name = "full_selfplay_grabs_v4"
        n_envs = "8"
        n_steps = "256"
        batch_size = "512"
        device = "auto"
        initial_checkpoint = "checkpoints/sim_ppo_selfplay_mechanics_v3.zip"
        initial_pool_dir = "checkpoints/full_selfplay_mechanics_v3"
    elif args.target == "train-grabs-fast":
        full_self_play = True
        run_name = "full_selfplay_grabs_v4_fast"
        checkpoint_name = "full_selfplay_grabs_v4_fast"
        final_name = "sim_ppo_selfplay_grabs_v4_fast.zip"
        log_name = "full_selfplay_grabs_v4_fast"
        n_envs = "16"
        n_steps = "256"
        batch_size = "1024"
        device = "cpu"
        initial_checkpoint = "checkpoints/sim_ppo_selfplay_mechanics_v3.zip"
        initial_pool_dir = "checkpoints/full_selfplay_mechanics_v3"
    elif args.target == "train-winloss-fast":
        full_self_play = True
        run_name = "full_selfplay_winloss_v5_fast"
        checkpoint_name = "full_selfplay_winloss_v5_fast"
        final_name = "sim_ppo_selfplay_winloss_v5_fast.zip"
        log_name = "full_selfplay_winloss_v5_fast"
        n_envs = "16"
        n_steps = "256"
        batch_size = "1024"
        device = "cpu"
        previous_pool = ROOT / "checkpoints" / "full_selfplay_grabs_v4_fast"
        latest_grab_checkpoint = latest_complete_checkpoint(previous_pool)
        initial_checkpoint = command_path(latest_grab_checkpoint) if latest_grab_checkpoint else "checkpoints/sim_ppo_selfplay_mechanics_v3.zip"
        initial_pool_dir = "checkpoints/full_selfplay_grabs_v4_fast" if latest_grab_checkpoint else "checkpoints/full_selfplay_mechanics_v3"
    elif args.target == "train-anti-stall-fast":
        full_self_play = True
        run_name = "full_selfplay_anti_stall_v6_fast"
        checkpoint_name = "full_selfplay_anti_stall_v6_fast"
        final_name = "sim_ppo_selfplay_anti_stall_v6_fast.zip"
        log_name = "full_selfplay_anti_stall_v6_fast"
        n_envs = "16"
        n_steps = "256"
        batch_size = "1024"
        device = "cpu"
        previous_pool = ROOT / "checkpoints" / "full_selfplay_grabs_v4_fast"
        latest_grab_checkpoint = latest_complete_checkpoint(previous_pool)
        initial_checkpoint = command_path(latest_grab_checkpoint) if latest_grab_checkpoint else "checkpoints/sim_ppo_selfplay_mechanics_v3.zip"
        initial_pool_dir = "checkpoints/full_selfplay_grabs_v4_fast" if latest_grab_checkpoint else "checkpoints/full_selfplay_mechanics_v3"
    elif args.target == "train-stalemate-fast":
        full_self_play = True
        run_name = "full_selfplay_stalemate_v7_fast"
        checkpoint_name = "full_selfplay_stalemate_v7_fast"
        final_name = "sim_ppo_selfplay_stalemate_v7_fast.zip"
        log_name = "full_selfplay_stalemate_v7_fast"
        n_envs = "16"
        n_steps = "256"
        batch_size = "1024"
        device = "cpu"
        previous_pool = ROOT / "checkpoints" / "full_selfplay_grabs_v4_fast"
        latest_grab_checkpoint = latest_complete_checkpoint(previous_pool)
        initial_checkpoint = command_path(latest_grab_checkpoint) if latest_grab_checkpoint else "checkpoints/sim_ppo_selfplay_mechanics_v3.zip"
        initial_pool_dir = "checkpoints/full_selfplay_grabs_v4_fast" if latest_grab_checkpoint else "checkpoints/full_selfplay_mechanics_v3"
        initial_ratings = None
        scripted_sample_rate = "0.0"
        best_checkpoint_rate = "0.25"
        latest_checkpoint_rate = "0.75"
    elif args.target == "train-mixed-fast":
        full_self_play = False
        run_name = "mixed_curriculum_v11_lateral_stall_fix_fast"
        checkpoint_name = "mixed_curriculum_v11_lateral_stall_fix_fast"
        final_name = "sim_ppo_mixed_curriculum_v11_lateral_stall_fix_fast.zip"
        log_name = "mixed_curriculum_v11_lateral_stall_fix_fast"
        n_envs = "16"
        n_steps = "256"
        batch_size = "1024"
        device = "cpu"
        latest_v9_checkpoint = latest_complete_checkpoint(ROOT / "checkpoints" / "mixed_curriculum_v9_clean_fast")
        previous_pool = ROOT / "checkpoints" / "full_selfplay_stalemate_v7_fast"
        latest_v7_checkpoint = latest_complete_checkpoint(previous_pool)
        if latest_v9_checkpoint is not None:
            initial_checkpoint = command_path(latest_v9_checkpoint)
        elif latest_v7_checkpoint is not None:
            initial_checkpoint = command_path(latest_v7_checkpoint)
        else:
            initial_checkpoint = "checkpoints/full_selfplay_grabs_v4_fast/iter_004.zip"
        initial_pool_dir = "checkpoints/full_selfplay_stalemate_v7_fast" if latest_v7_checkpoint else "checkpoints/full_selfplay_grabs_v4_fast"
        initial_ratings = "runs/full_selfplay_stalemate_v7_fast/metrics.json" if (ROOT / "runs" / "full_selfplay_stalemate_v7_fast" / "metrics.json").exists() else None
        scripted_sample_rate = "0.34"
        best_checkpoint_rate = "0.50"
        latest_checkpoint_rate = "0.50"

    if args.target in {"train-grabs", "train-grabs-fast", "train-winloss-fast", "train-anti-stall-fast", "train-stalemate-fast", "train-mixed-fast"}:
        if args.target != "train-mixed-fast":
            initial_ratings = None
            scripted_sample_rate = "0.0"
            best_checkpoint_rate = "0.25"
            latest_checkpoint_rate = "0.75"
        (ROOT / "runs" / "full_selfplay_grabs_v4").mkdir(parents=True, exist_ok=True)
        (ROOT / "checkpoints" / "full_selfplay_grabs_v4").mkdir(parents=True, exist_ok=True)
        (ROOT / "runs" / run_name).mkdir(parents=True, exist_ok=True)
        (ROOT / "checkpoints" / checkpoint_name).mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON),
            "scripts/train_sim_ppo_selfplay.py",
            "--iterations",
            "20",
            "--timesteps-per-iteration",
            "32768",
            "--bootstrap-iterations",
            "0",
            "--scripted-sample-rate",
            scripted_sample_rate,
            "--elo-sampling",
            "--latest-checkpoint-rate",
            latest_checkpoint_rate,
            "--best-checkpoint-rate",
            best_checkpoint_rate,
            "--old-sample-rate",
            "0.0",
            "--n-envs",
            n_envs,
            "--n-steps",
            n_steps,
            "--batch-size",
            batch_size,
            "--device",
            device,
            "--initial-checkpoint",
            initial_checkpoint,
            "--initial-pool-dir",
            initial_pool_dir,
            "--checkpoint-dir",
            f"checkpoints/{checkpoint_name}",
            "--final-checkpoint",
            f"checkpoints/{final_name}",
            "--run-dir",
            f"runs/{run_name}",
        ]
        if full_self_play:
            command.insert(command.index("--bootstrap-iterations"), "--full-self-play")
        if initial_ratings is not None:
            command.extend(["--initial-ratings", initial_ratings])
        pid = launch(
            command,
            stdout=logs / f"{log_name}.out.log",
            stderr=logs / f"{log_name}.err.log",
            hidden=True,
        )
        print(f"TRAIN_PID={pid}")
        return 0

    if args.target == "live-old":
        pid = launch(
            [
                str(PYTHON),
                "scripts/live_play.py",
                "--agent",
                "checkpoint",
                "--checkpoint",
                "checkpoints/full_selfplay_movement_v2/iter_015.zip",
                "--screen-config",
                "config/live_screen.example.yaml",
                "--quit-hotkey",
                "f10",
                "--tap-seconds",
                "0.14",
                "--max-non-attack-streak",
                "4",
            ],
            stdout=logs / "live_old_checkpoint.out.log",
            stderr=logs / "live_old_checkpoint.err.log",
            hidden=False,
        )
        print(f"LIVE_PID={pid}")
        return 0

    if args.target == "live-new":
        pid = launch(
            [
                str(PYTHON),
                "scripts/live_play.py",
                "--agent",
                "checkpoint",
                "--checkpoint",
                "checkpoints/sim_ppo_selfplay_mechanics_v3.zip",
                "--screen-config",
                "config/live_screen.example.yaml",
                "--quit-hotkey",
                "f10",
                "--tap-seconds",
                "0.14",
                "--max-non-attack-streak",
                "4",
            ],
            stdout=logs / "live_new_checkpoint.out.log",
            stderr=logs / "live_new_checkpoint.err.log",
            hidden=False,
        )
        print(f"LIVE_PID={pid}")
        return 0

    follow_dir = "checkpoints/full_selfplay_mechanics_v3"
    checkpoint_dir = ROOT / follow_dir
    if args.target == "visualizer-grabs":
        follow_dir = "checkpoints/full_selfplay_grabs_v4"
        checkpoint_dir = ROOT / follow_dir
    elif args.target == "visualizer-grabs-fast":
        follow_dir = "checkpoints/full_selfplay_grabs_v4_fast"
        checkpoint_dir = ROOT / follow_dir
    elif args.target == "visualizer-winloss-fast":
        follow_dir = "checkpoints/full_selfplay_winloss_v5_fast"
        checkpoint_dir = ROOT / follow_dir
    elif args.target == "visualizer-anti-stall-fast":
        follow_dir = "checkpoints/full_selfplay_anti_stall_v6_fast"
        checkpoint_dir = ROOT / follow_dir
    elif args.target == "visualizer-stalemate-fast":
        follow_dir = "checkpoints/full_selfplay_stalemate_v7_fast"
        checkpoint_dir = ROOT / follow_dir
    elif args.target == "visualizer-mixed-fast":
        follow_dir = "checkpoints/mixed_curriculum_v11_lateral_stall_fix_fast"
        checkpoint_dir = ROOT / follow_dir
    checkpoint_args = ["--p1", "scripted", "--p1-scripted", "rushdown", "--p2", "scripted", "--p2-scripted", "poke"]
    if args.target == "visualizer-mixed-fast" and any(checkpoint_dir.glob("*.zip")):
        checkpoint_args = [
            "--p1",
            "checkpoint",
            "--follow-dir",
            follow_dir,
            "--p2",
            "scripted",
            "--p2-scripted",
            "rushdown",
        ]
    elif any(checkpoint_dir.glob("*.zip")):
        checkpoint_args = [
            "--p1",
            "checkpoint",
            "--follow-dir",
            follow_dir,
            "--p2",
            "checkpoint",
            "--p2-follow-dir",
            follow_dir,
            "--p2-follow-previous",
        ]
    pid = launch(
        [
            str(PYTHON),
            "scripts/visualize_sim.py",
            *checkpoint_args,
            "--speed",
            "4",
        ],
        stdout=logs / "visualizer_mechanics_v3.out.log",
        stderr=logs / "visualizer_mechanics_v3.err.log",
        hidden=False,
    )
    print(f"VISUALIZER_PID={pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

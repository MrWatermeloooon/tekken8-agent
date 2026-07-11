from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path


FPS_RE = re.compile(r"\|\s+fps\s+\|\s+([0-9.]+)\s+\|")
ITERATION_RE = re.compile(
    r"iteration=(?P<iteration>\d+)\s+checkpoint=(?P<checkpoint>\S+)\s+pool_size=(?P<pool_size>\d+)\s+"
    r"n_envs=(?P<n_envs>\d+)\s+scripted_sample_rate=(?P<scripted>[0-9.]+)\s+"
    r"best_checkpoint_rate=(?P<best>[0-9.]+)\s+latest_checkpoint_rate=(?P<latest>[0-9.]+)\s+"
    r"scripted_win_rate=(?P<scripted_win>[0-9.]+)\s+scripted_reward=(?P<scripted_reward>-?[0-9.]+)\s+"
    r"checkpoint_win_rate=(?P<checkpoint_win>\S+)"
)


def tail_text(path: Path, max_bytes: int = 64_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def latest_checkpoint(checkpoint_dir: Path) -> str:
    checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        return "none"
    path = checkpoints[-1]
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
    return f"{path.name}@{stamp}"


def last_match(pattern: re.Pattern[str], text: str) -> str:
    matches = list(pattern.finditer(text))
    if not matches:
        return "none"
    match = matches[-1]
    if pattern is FPS_RE:
        return match.group(1)
    groups = match.groupdict()
    return (
        f"iter={groups['iteration']} pool={groups['pool_size']} scripted={groups['scripted']} "
        f"best={groups['best']} latest={groups['latest']} scripted_win={groups['scripted_win']} "
        f"scripted_reward={groups['scripted_reward']} checkpoint_win={groups['checkpoint_win']}"
    )


def write_status_line(out_path: Path, checkpoint_dir: Path, train_log: Path, err_log: Path) -> None:
    text = tail_text(train_log)
    err_size = err_log.stat().st_size if err_log.exists() else 0
    train_size = train_log.stat().st_size if train_log.exists() else 0
    line = (
        f"{datetime.now().isoformat(timespec='seconds')} "
        f"latest_checkpoint={latest_checkpoint(checkpoint_dir)} "
        f"last_fps={last_match(FPS_RE, text)} "
        f"last_iteration='{last_match(ITERATION_RE, text)}' "
        f"train_log_bytes={train_size} err_log_bytes={err_size}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write compact status lines for a long-running training log.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--err-log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    train_log = Path(args.train_log)
    err_log = Path(args.err_log)
    out_path = Path(args.out)

    while True:
        write_status_line(out_path, checkpoint_dir, train_log, err_log)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

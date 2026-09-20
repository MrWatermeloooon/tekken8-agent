from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from tkinter import BOTH, BOTTOM, Canvas, Label, Tk


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEED = REPO_ROOT / "build" / "Release" / "t8_v2_visualizer_feed.exe"
DEFAULT_PROFILES = REPO_ROOT / "data" / "generated" / "opponent_profiles.csv"
DEFAULT_MOVES = REPO_ROOT / "data" / "generated" / "character_move_specs.csv"


def _load_profiles(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_profile(args: argparse.Namespace, profiles: list[dict[str, str]]) -> tuple[int, dict[str, str]]:
    if args.profile_index is not None:
        if args.profile_index < 0 or args.profile_index >= len(profiles):
            raise ValueError(f"--profile-index must be between 0 and {len(profiles) - 1}")
        return args.profile_index, profiles[args.profile_index]
    matches = [
        (index, profile)
        for index, profile in enumerate(profiles)
        if profile["character_slug"] == args.opponent_character
        and profile["archetype"] == args.opponent_archetype
        and int(profile["variation_id"]) == args.variation
    ]
    if not matches:
        raise ValueError(
            "no generated profile matches "
            f"{args.opponent_character}/{args.opponent_archetype}/variation {args.variation}"
        )
    return matches[0]


def _latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = [path for path in directory.glob("*.t8ppo") if path.is_file()]
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns) if checkpoints else None


def _feed_command(
    args: argparse.Namespace,
    profile_index: int,
    checkpoint: Path | None,
    *,
    steps: int = 0,
) -> list[str]:
    command = [
        str(args.feed),
        "--profile-index",
        str(profile_index),
        "--opponent-catalog",
        str(args.opponent_catalog),
        "--character-moves",
        str(args.character_moves),
        "--observation-mode",
        args.observation_mode,
        "--seed",
        str(args.seed),
        "--steps",
        str(steps),
        "--interval-ms",
        "0" if steps else str(max(1, round(1000 / (15 * args.speed)))),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    return command


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


def run_headless(args: argparse.Namespace, profile_index: int, checkpoint: Path | None) -> int:
    command = _feed_command(args, profile_index, checkpoint, steps=args.headless_steps)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        creationflags=_creation_flags(),
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"visualizer feed exited {completed.returncode}")
    states = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(states) != args.headless_steps:
        raise RuntimeError(f"expected {args.headless_steps} feed states, received {len(states)}")
    last = states[-1]
    print(
        f"headless_ok states={len(states)} episode={last['episode']} frame={last['frame']} "
        f"p1_hp={last['p1']['health']:.1f} p2_hp={last['p2']['health']:.1f}"
    )
    return 0


class V2Visualizer:
    def __init__(
        self,
        args: argparse.Namespace,
        profile_index: int,
        profile: dict[str, str],
        checkpoint: Path | None,
    ) -> None:
        self.args = args
        self.profile_index = profile_index
        self.profile = profile
        self.checkpoint = checkpoint
        self.width = args.width
        self.height = args.height
        self.paused = False
        self.state: dict | None = None
        self.feed_error = ""
        self.process: subprocess.Popen[str] | None = None
        self.generation = 0
        self.states: queue.Queue[tuple[int, dict]] = queue.Queue(maxsize=4)
        self.last_follow_check = 0.0

        self.root = Tk()
        self.root.title("Tekken 8 Agent V2 - CUDA Simulator Visualizer")
        self.canvas = Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg="#0b1016",
            highlightthickness=0,
        )
        self.canvas.pack(fill=BOTH, expand=True)
        self.status = Label(
            self.root,
            text="Space: pause | R: reset feed | +/-: speed",
            anchor="w",
            bg="#101820",
            fg="#d8e1e8",
        )
        self.status.pack(side=BOTTOM, fill="x")
        self.root.bind("<space>", lambda _event: self.toggle_pause())
        self.root.bind("r", lambda _event: self.restart_feed())
        self.root.bind("R", lambda _event: self.restart_feed())
        self.root.bind("+", lambda _event: self.change_speed(1))
        self.root.bind("=", lambda _event: self.change_speed(1))
        self.root.bind("-", lambda _event: self.change_speed(-1))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.start_feed()

    def run(self) -> None:
        self.root.after(20, self.tick)
        self.root.mainloop()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def change_speed(self, delta: int) -> None:
        self.args.speed = max(0.25, min(8.0, self.args.speed * (1.5 if delta > 0 else 2.0 / 3.0)))
        self.restart_feed()

    def restart_feed(self) -> None:
        self.stop_feed()
        self.start_feed()

    def start_feed(self) -> None:
        if self.args.follow_dir is not None:
            followed = _latest_checkpoint(self.args.follow_dir)
            if followed is not None:
                self.checkpoint = followed
        command = _feed_command(self.args, self.profile_index, self.checkpoint)
        self.generation += 1
        generation = self.generation
        self.feed_error = ""
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=_creation_flags(),
        )
        threading.Thread(target=self._read_feed, args=(generation, self.process), daemon=True).start()

    def stop_feed(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _read_feed(self, generation: int, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                state = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = (generation, state)
            while True:
                try:
                    self.states.put_nowait(item)
                    break
                except queue.Full:
                    try:
                        self.states.get_nowait()
                    except queue.Empty:
                        pass
        return_code = process.wait()
        if return_code != 0 and generation == self.generation:
            assert process.stderr is not None
            self.feed_error = process.stderr.read().strip() or f"feed exited {return_code}"

    def _maybe_follow_checkpoint(self) -> None:
        if self.args.follow_dir is None:
            return
        now = time.monotonic()
        if now - self.last_follow_check < self.args.follow_check_seconds:
            return
        self.last_follow_check = now
        latest = _latest_checkpoint(self.args.follow_dir)
        if latest is not None and latest != self.checkpoint:
            self.checkpoint = latest
            self.restart_feed()

    def tick(self) -> None:
        self._maybe_follow_checkpoint()
        if not self.paused:
            while True:
                try:
                    generation, state = self.states.get_nowait()
                except queue.Empty:
                    break
                if generation == self.generation:
                    self.state = state
        self.draw()
        self.root.after(33, self.tick)

    def draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        canvas.create_rectangle(0, 0, self.width, self.height, fill="#0b1016", outline="")
        canvas.create_rectangle(0, 0, self.width, 136, fill="#111b25", outline="")
        if self.state is None:
            message = self.feed_error or "Starting CUDA simulator feed..."
            canvas.create_text(
                self.width / 2,
                self.height / 2,
                text=message,
                fill="#f6e05e" if self.feed_error else "#d8e1e8",
                font=("Segoe UI", 16, "bold"),
                width=self.width - 100,
            )
            return
        self._draw_health_bars()
        self._draw_stage()
        self._draw_fighter("p1", "#68d391", "Jun")
        self._draw_fighter("p2", "#f687b3", self.profile["character_slug"].replace("_", " ").title())
        self._draw_text()

    def _draw_health_bars(self) -> None:
        assert self.state is not None
        margin, bar_width, bar_height, y = 42, 430, 24, 34
        self._health_bar(margin, y, bar_width, bar_height, self.state["p1"]["health"], "#68d391", False)
        self._health_bar(
            self.width - margin - bar_width,
            y,
            bar_width,
            bar_height,
            self.state["p2"]["health"],
            "#f687b3",
            True,
        )
        remaining = max(0, self.state["max_frames"] - self.state["frame"])
        timer = int(remaining / 60)
        self.canvas.create_text(
            self.width / 2, y + 12, text=str(timer), fill="#f7fafc", font=("Segoe UI", 26, "bold")
        )

    def _health_bar(self, x: float, y: float, width: float, height: float, health: float, color: str, right: bool) -> None:
        assert self.state is not None
        ratio = max(0.0, min(1.0, health / self.state["max_health"]))
        self.canvas.create_rectangle(x, y, x + width, y + height, fill="#2d3748", outline="#607080")
        fill_width = width * ratio
        start = x + width - fill_width if right else x
        self.canvas.create_rectangle(start, y, start + fill_width, y + height, fill=color, outline="")

    def _draw_stage(self) -> None:
        assert self.state is not None
        floor = self._floor_y()
        left = self._world_x(-self.state["stage_half_width"])
        right = self._world_x(self.state["stage_half_width"])
        self.canvas.create_line(left, floor, right, floor, fill="#7f8ea3", width=4)
        self.canvas.create_line(left, floor - 155, left, floor + 18, fill="#e2e8f0", width=3)
        self.canvas.create_line(right, floor - 155, right, floor + 18, fill="#e2e8f0", width=3)
        for index in range(9):
            x = left + (right - left) * index / 8
            self.canvas.create_line(x, floor - 8, x, floor + 8, fill="#4a5568")

    def _draw_fighter(self, key: str, color: str, label: str) -> None:
        assert self.state is not None
        fighter = self.state[key]
        outline = "#edf2f7"
        if fighter["hitstun"] > 0:
            outline = "#f6e05e"
        elif fighter["blockstun"] > 0 or fighter["guard"] != 0:
            outline = "#63b3ed"
        x = self._world_x(fighter["x"])
        floor = self._floor_y() - min(50, fighter["airborne"] * 2)
        self.canvas.create_rectangle(x - 21, floor - 118, x + 21, floor, fill=color, outline=outline, width=3)
        self.canvas.create_oval(x - 18, floor - 154, x + 18, floor - 118, fill=color, outline=outline, width=3)
        self.canvas.create_text(x, self._floor_y() + 28, text=label, fill="#e2e8f0", font=("Segoe UI", 12, "bold"))

    def _draw_text(self) -> None:
        assert self.state is not None
        checkpoint = self.checkpoint.name if self.checkpoint else "scripted learner"
        lines = [
            f"Episode {self.state['episode']} | Frame {self.state['frame']} | Speed {self.args.speed:.2f}x | Reward {self.state['reward_p1']:.2f}",
            f"P1 {checkpoint}: {self.state['p1_action']} | move {self.state['p1']['move']} | hitstun {self.state['p1']['hitstun']} | blockstun {self.state['p1']['blockstun']}",
            f"P2 {self.profile['name']}: {self.state['p2_action']} | move {self.state['p2']['move']} | hitstun {self.state['p2']['hitstun']} | blockstun {self.state['p2']['blockstun']}",
            f"Distance {self.state['distance']:.2f} | P1 whiffs {self.state['p1']['whiffs']} | P2 whiffs {self.state['p2']['whiffs']}",
        ]
        if self.state["round_over"]:
            winner = "draw" if self.state["winner"] == 0 else f"P{self.state['winner']} wins"
            lines.append(f"ROUND OVER: {winner}")
        for index, line in enumerate(lines):
            self.canvas.create_text(42, 82 + index * 22, text=line, anchor="w", fill="#d8e1e8", font=("Consolas", 11))
        if self.paused:
            self.canvas.create_text(
                self.width / 2, 205, text="PAUSED", fill="#f6e05e", font=("Segoe UI", 28, "bold")
            )

    def _world_x(self, x: float) -> float:
        assert self.state is not None
        half_width = self.state["stage_half_width"]
        normalized = (x + half_width) / (half_width * 2)
        return 84 + normalized * (self.width - 168)

    def _floor_y(self) -> float:
        return self.height - 118

    def close(self) -> None:
        self.stop_feed()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch the native V2 CUDA simulator in a Tkinter window.")
    parser.add_argument("--feed", type=Path, default=DEFAULT_FEED)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--follow-dir", type=Path, default=None, help="Reload the newest .t8ppo checkpoint in this directory.")
    parser.add_argument("--follow-check-seconds", type=float, default=2.0)
    parser.add_argument("--observation-mode", choices=["visual", "privileged"], default="visual")
    parser.add_argument("--opponent-catalog", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--character-moves", type=Path, default=DEFAULT_MOVES)
    parser.add_argument("--profile-index", type=int, default=None)
    parser.add_argument("--opponent-character", default="reina")
    parser.add_argument("--opponent-archetype", default="rushdown")
    parser.add_argument("--variation", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=620)
    parser.add_argument("--headless-steps", type=int, default=0)
    args = parser.parse_args()
    args.feed = args.feed.resolve()
    args.opponent_catalog = args.opponent_catalog.resolve()
    args.character_moves = args.character_moves.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if args.follow_dir is not None:
        args.follow_dir = args.follow_dir.resolve()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.headless_steps < 0:
        parser.error("--headless-steps cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    if not args.feed.exists():
        raise FileNotFoundError(f"build the V2 visualizer feed first: {args.feed}")
    profiles = _load_profiles(args.opponent_catalog)
    profile_index, profile = _resolve_profile(args, profiles)
    checkpoint = args.checkpoint
    if args.follow_dir is not None:
        checkpoint = _latest_checkpoint(args.follow_dir)
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.headless_steps:
        return run_headless(args, profile_index, checkpoint)
    V2Visualizer(args, profile_index, profile, checkpoint).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

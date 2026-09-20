from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from tkinter import BOTH, BOTTOM, Canvas, Label, Tk


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEED = REPO_ROOT / "build" / "Release" / "t8_v2_visualizer_feed.exe"
DEFAULT_PROFILES = REPO_ROOT / "data" / "generated" / "opponent_profiles.csv"
DEFAULT_MOVES = REPO_ROOT / "data" / "generated" / "character_move_specs.csv"

# Mirrors t8::v2::CharacterGroup in include/t8_v2/roster.hpp.
GROUP_FUNDAMENTALS = 1 << 0
GROUP_RUSHDOWN = 1 << 1
GROUP_STANCE_HEAVY = 1 << 2
GROUP_GRAPPLER = 1 << 3
GROUP_KEEP_OUT = 1 << 4
GROUP_EVASIVE = 1 << 5
GROUP_SPECIALIST = 1 << 6

# Mirrors the CharacterGroups rotation in curriculum_for_update() in src/train.cpp.
CHARACTER_GROUP_ROTATION = [
    GROUP_FUNDAMENTALS, GROUP_RUSHDOWN, GROUP_STANCE_HEAVY,
    GROUP_GRAPPLER, GROUP_KEEP_OUT, GROUP_EVASIVE, GROUP_SPECIALIST,
]

# Neither launch script overrides --curriculum-updates, so this matches the
# trainer's own default. It only affects which representative opponent this
# viewer picks to display; it never touches the actual training run.
DEFAULT_CURRICULUM_UPDATES = 100

_SELF_PLAY_RE = re.compile(r"self_play=\d+%latest:(\d+)/\d+%best:(\d+)")
_KV_RE = re.compile(r"(\w+)=(-?[\w.+-]+)")


def _curriculum_stage(update: int, curriculum_updates: int = DEFAULT_CURRICULUM_UPDATES) -> tuple[str, int]:
    """Mirrors curriculum_for_update() in src/train.cpp for auto-curriculum runs."""
    stage_span = max(1, (curriculum_updates + 3) // 4)
    if update <= stage_span:
        return "jun_fundamentals", 0
    if update <= 2 * stage_span:
        group = CHARACTER_GROUP_ROTATION[(update - stage_span - 1) % len(CHARACTER_GROUP_ROTATION)]
        return "character_groups", group
    if update <= 3 * stage_span:
        return "full_roster", 0
    return "adversarial_league", 0


def _parse_training_line(line: str) -> dict[str, str]:
    return dict(_KV_RE.findall(line))


def read_training_status(run_dir: Path, tail_lines: int = 200) -> dict:
    """Reads the trainer's own stdout log to determine what it is actually
    training against right now: curriculum stage, self-play opponents (if
    any), current update, and the most recent measured entropy/win rate.
    This never touches the running trainer process, only its log file.
    """
    log_path = run_dir / "trainer.stdout.log"
    status: dict = {
        "update": 0,
        "entropy": None,
        "self_play": None,
        "eval_win_rate": None,
        "stochastic_eval_win_rate": None,
    }
    if not log_path.exists():
        return status
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()[-tail_lines:]
    except OSError:
        return status
    for line in lines:
        if not line.startswith("update="):
            continue
        fields = _parse_training_line(line)
        update_field = fields.get("update", "0/0")
        try:
            status["update"] = int(update_field.split("/", 1)[0])
        except ValueError:
            continue
        if "entropy" in fields:
            try:
                status["entropy"] = float(fields["entropy"])
            except ValueError:
                pass
        match = _SELF_PLAY_RE.search(line)
        if match:
            status["self_play"] = {"latest": int(match.group(1)), "best": int(match.group(2))}
        else:
            status["self_play"] = None
        if "eval_win_rate" in fields:
            try:
                status["eval_win_rate"] = float(fields["eval_win_rate"])
                status["stochastic_eval_win_rate"] = float(fields.get("stochastic_eval_win_rate", "nan"))
            except ValueError:
                pass
    return status


def _load_profiles(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _eligible_stage_profiles(
    profiles: list[dict[str, str]], stage: str, group_mask: int
) -> list[tuple[int, dict[str, str]]]:
    """Mirrors MatchupScheduler::eligible_profile_indices() in src/roster.cpp
    closely enough to pick a genuinely representative opponent for display,
    without reimplementing its priority-weighted sampling.
    """
    if stage == "jun_fundamentals":
        return [
            (index, profile) for index, profile in enumerate(profiles)
            if int(profile["group_mask"]) & GROUP_FUNDAMENTALS
            and int(profile["archetype_id"]) <= 4
            and int(profile["variation_id"]) <= 1
        ]
    if stage == "character_groups":
        return [
            (index, profile) for index, profile in enumerate(profiles)
            if int(profile["group_mask"]) & group_mask
        ]
    return list(enumerate(profiles))


def _pick_stage_profile(
    profiles: list[dict[str, str]], update: int, curriculum_updates: int = DEFAULT_CURRICULUM_UPDATES
) -> tuple[int, dict[str, str]]:
    stage, group_mask = _curriculum_stage(update, curriculum_updates)
    eligible = _eligible_stage_profiles(profiles, stage, group_mask)
    if not eligible:
        return 0, profiles[0]
    # Deterministic on (stage, update) so the displayed opponent stays put
    # between polls instead of flickering every couple of seconds, but still
    # varies across the roster and rotates as the stage advances.
    return eligible[update % len(eligible)]


def _checkpoint_for_update(checkpoints_dir: Path, update: int) -> Path | None:
    candidate = checkpoints_dir / f"update_{update}.t8ppo"
    return candidate if candidate.is_file() else None


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
    opponent_checkpoint: Path | None = None,
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
    if opponent_checkpoint is not None:
        command.extend(["--opponent-checkpoint", str(opponent_checkpoint)])
    return command


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


def run_headless(
    args: argparse.Namespace,
    profile_index: int,
    checkpoint: Path | None,
    *,
    opponent_checkpoint: Path | None = None,
) -> int:
    command = _feed_command(
        args, profile_index, checkpoint, steps=args.headless_steps, opponent_checkpoint=opponent_checkpoint)
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
    STATUS_BAND_HEIGHT = 34

    STAGE_ACCENTS = {
        "jun_fundamentals": ("#63b3ed", "FUNDAMENTALS"),
        "character_groups": ("#b794f4", "CHARACTER GROUPS"),
        "full_roster": ("#f6ad55", "FULL ROSTER"),
        "adversarial_league": ("#fc8181", "SELF-PLAY"),
        "unknown": ("#718096", "UNKNOWN"),
    }

    def __init__(
        self,
        args: argparse.Namespace,
        profile_index: int,
        profile: dict[str, str],
        checkpoint: Path | None,
        opponent_checkpoint: Path | None = None,
        training_status: dict | None = None,
    ) -> None:
        self.args = args
        self.profile_index = profile_index
        self.profile = profile
        self.checkpoint = checkpoint
        self.opponent_checkpoint = opponent_checkpoint
        self.training_status: dict = training_status or {}
        self.width = args.width
        self.height = args.height
        self.paused = False
        self.state: dict | None = None
        self.feed_error = ""
        self.process: subprocess.Popen[str] | None = None
        self.generation = 0
        self.states: queue.Queue[tuple[int, dict]] = queue.Queue(maxsize=4)
        self.last_follow_check = 0.0
        self._profiles_for_run_dir: list[dict[str, str]] | None = None

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
        command = _feed_command(
            self.args, self.profile_index, self.checkpoint,
            opponent_checkpoint=self.opponent_checkpoint,
        )
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
        if self.args.follow_run_dir is not None:
            self._maybe_follow_run_dir()
            return
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

    def _maybe_follow_run_dir(self) -> None:
        """Reads what the live trainer is actually doing (curriculum stage,
        self-play opponents, entropy, win rate) and keeps the displayed fight
        matching it. This only reads files the trainer writes; it never
        touches the running training process.
        """
        now = time.monotonic()
        if now - self.last_follow_check < self.args.follow_check_seconds:
            return
        self.last_follow_check = now
        run_dir = self.args.follow_run_dir
        self.training_status = read_training_status(run_dir)
        checkpoints_dir = run_dir / "checkpoints"
        learner_checkpoint = _latest_checkpoint(checkpoints_dir)
        if learner_checkpoint is None:
            return
        self_play = self.training_status.get("self_play")
        if self_play is not None:
            opponent_checkpoint = _checkpoint_for_update(checkpoints_dir, self_play["latest"])
            profile_index, profile = self.profile_index, self.profile
        else:
            opponent_checkpoint = None
            if self._profiles_for_run_dir is None:
                self._profiles_for_run_dir = _load_profiles(self.args.opponent_catalog)
            profile_index, profile = _pick_stage_profile(
                self._profiles_for_run_dir, self.training_status.get("update", 0))
        changed = (
            learner_checkpoint != self.checkpoint
            or opponent_checkpoint != self.opponent_checkpoint
            or profile_index != self.profile_index
        )
        self.checkpoint = learner_checkpoint
        self.opponent_checkpoint = opponent_checkpoint
        self.profile_index = profile_index
        self.profile = profile
        if changed:
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

    def _header_offset(self) -> int:
        """Height reserved at the top for the training-status band. Keeping
        this as a single source of truth is what keeps the health bars, the
        fight-info text, and the status band from ever overlapping."""
        return self.STATUS_BAND_HEIGHT if self.args.follow_run_dir is not None else 0

    def draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        canvas.create_rectangle(0, 0, self.width, self.height, fill="#0b1016", outline="")
        offset = self._header_offset()
        if self.args.follow_run_dir is not None:
            self._draw_training_status()
        canvas.create_rectangle(0, offset, self.width, offset + 136, fill="#111b25", outline="")
        if self.state is None:
            message = self.feed_error or "Starting CUDA simulator feed..."
            canvas.create_text(
                self.width / 2,
                offset + (self.height - offset) / 2,
                text=message,
                fill="#f6e05e" if self.feed_error else "#d8e1e8",
                font=("Segoe UI", 16, "bold"),
                width=self.width - 100,
            )
            return
        self._draw_health_bars(offset)
        self._draw_stage()
        self._draw_fighter("p1", "#68d391", "Jun")
        self._draw_fighter("p2", "#f687b3", self._p2_label())
        self._draw_text(offset)

    def _p2_label(self) -> str:
        if self.opponent_checkpoint is not None:
            return f"Self-play ({self.opponent_checkpoint.stem})"
        return self.profile["character_slug"].replace("_", " ").title()

    def _draw_health_bars(self, offset: int) -> None:
        assert self.state is not None
        margin, bar_width, bar_height, y = 42, 430, 24, offset + 34
        self.canvas.create_text(
            margin, y - 14, text="JUN", anchor="w", fill="#9ae6b4", font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(
            self.width - margin, y - 14, text=self._p2_label().upper(), anchor="e",
            fill="#fbb6ce", font=("Segoe UI", 10, "bold"))
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
        canvas = self.canvas
        floor = self._floor_y()
        left = self._world_x(-self.state["stage_half_width"])
        right = self._world_x(self.state["stage_half_width"])
        top = floor - 230

        # Backdrop: a few stacked bands instead of a flat void, so the arena
        # reads as an enclosed stage rather than empty space.
        band_colors = ["#0c1220", "#101a2c", "#152439", "#1b3049", "#213c58"]
        band_height = (floor - top) / len(band_colors)
        for index, color in enumerate(band_colors):
            y0 = top + band_height * index
            canvas.create_rectangle(0, y0, self.width, y0 + band_height + 1, fill=color, outline="")

        # Distant pillars for depth, sitting behind the floor and fighters.
        for fraction in (0.06, 0.20, 0.80, 0.94):
            px = left + (right - left) * fraction
            canvas.create_rectangle(px - 11, top + 30, px + 11, floor, fill="#182740", outline="#22314a")

        # Floor plane with a perspective taper toward a horizon line, plus a
        # converging tile grid, instead of one flat ground line.
        inset = (right - left) * 0.17
        horizon = floor - 50
        canvas.create_polygon(
            left, floor, right, floor, right - inset, horizon, left + inset, horizon,
            fill="#253146", outline="#3a4863",
        )
        for index in range(9):
            fraction = index / 8
            bx = left + (right - left) * fraction
            tx = left + inset + (right - left - 2 * inset) * fraction
            canvas.create_line(bx, floor, tx, horizon, fill="#3a4863")
        for depth in (0.4, 0.75):
            y = floor - (floor - horizon) * depth
            edge_inset = inset * depth
            canvas.create_line(left + edge_inset, y, right - edge_inset, y, fill="#334162")
        canvas.create_line(left, floor, right, floor, fill="#8a9bb5", width=2)

        # Stage-out boundary walls: a soft glowing barrier (stacked lines of
        # decreasing width, brightest in the middle) like a ring-out fence.
        for wall_x in (left, right):
            for width_px, color in ((10, "#1c4763"), (5, "#3f8fc0"), (2, "#cdeeff")):
                canvas.create_line(wall_x, floor - 190, wall_x, floor + 6, fill=color, width=width_px)

    @staticmethod
    def _shade(color: str, factor: float) -> str:
        value = color.lstrip("#")
        r, g, b = (int(value[index:index + 2], 16) for index in (0, 2, 4))
        r, g, b = (max(0, min(255, round(channel * factor))) for channel in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _facing_right(self, key: str) -> bool:
        assert self.state is not None
        other = "p2" if key == "p1" else "p1"
        return self.state[key]["x"] <= self.state[other]["x"]

    def _draw_fighter(self, key: str, color: str, label: str) -> None:
        assert self.state is not None
        fighter = self.state[key]
        attacking = fighter["move"] != "-"
        blocking = fighter["blockstun"] > 0 or fighter["guard"] != 0
        hit = fighter["hitstun"] > 0
        outline = "#f6e05e" if hit else "#63b3ed" if blocking else "#edf2f7"
        back_color = self._shade(color, 0.62)

        facing = 1 if self._facing_right(key) else -1
        cx = self._world_x(fighter["x"]) - facing * 8 * hit  # small recoil flinch on hit
        ground_y = self._floor_y() - min(50, fighter["airborne"] * 2)
        hip = (cx, ground_y - 58)
        shoulder = (cx + facing * 2, ground_y - 108)
        head_center = (cx + facing * 3, ground_y - 132)
        head_radius = 17

        canvas = self.canvas
        # Contact/ground shadow, drawn first so everything else sits on it.
        canvas.create_oval(cx - 24, self._floor_y() - 6, cx + 24, self._floor_y() + 5,
                            fill="#05070c", outline="")

        line = lambda p, q, w, c: canvas.create_line(*p, *q, width=w, fill=c, capstyle="round")

        # Back leg and back arm first so the torso and front limbs overlap
        # them, giving the silhouette a sense of depth/facing.
        line(hip, (cx - facing * 15, self._floor_y()), 15, back_color)
        line(shoulder, (cx - facing * 21, ground_y - 90), 10, back_color)

        line(hip, shoulder, 27, color)  # torso
        canvas.create_oval(
            head_center[0] - head_radius, head_center[1] - head_radius,
            head_center[0] + head_radius, head_center[1] + head_radius,
            fill=color, outline=outline, width=3,
        )

        front_foot = (cx + facing * 19, self._floor_y())
        line(hip, front_foot, 15, color)
        if blocking:
            guard_hand = (shoulder[0] + facing * 12, ground_y - 128)
            line(shoulder, guard_hand, 10, color)
        elif attacking:
            punch_hand = (cx + facing * 48, ground_y - 112)
            line(shoulder, punch_hand, 10, color)
        else:
            idle_hand = (cx + facing * 17, ground_y - 116)
            line(shoulder, idle_hand, 10, color)

        canvas.create_text(cx, self._floor_y() + 28, text=label, fill="#e2e8f0", font=("Segoe UI", 12, "bold"))

    def _draw_text(self, offset: int) -> None:
        assert self.state is not None
        checkpoint = self.checkpoint.name if self.checkpoint else "scripted learner"
        base_y = offset + 88
        self.canvas.create_text(
            42, base_y,
            text=(
                f"Episode {self.state['episode']}  |  Frame {self.state['frame']}  |  "
                f"Speed {self.args.speed:.2f}x  |  Reward {self.state['reward_p1']:+.2f}"
            ),
            anchor="w", fill="#a0aec0", font=("Consolas", 11),
        )
        fighter_lines = [
            ("P1", "#9ae6b4", checkpoint, self.state["p1"], self.state["p1_action"]),
            ("P2", "#fbb6ce", self._p2_label(), self.state["p2"], self.state["p2_action"]),
        ]
        for row, (tag, color, name, fighter, action) in enumerate(fighter_lines):
            y = base_y + 22 * (row + 1)
            self.canvas.create_text(42, y, text=tag, anchor="w", fill=color, font=("Consolas", 11, "bold"))
            self.canvas.create_text(
                72, y, anchor="w", fill="#d8e1e8", font=("Consolas", 11),
                text=(
                    f"{name}: {action}  |  move {fighter['move']}  |  "
                    f"hitstun {fighter['hitstun']}  |  blockstun {fighter['blockstun']}"
                ),
            )
        distance_y = base_y + 22 * 3
        self.canvas.create_text(
            42, distance_y, anchor="w", fill="#a0aec0", font=("Consolas", 11),
            text=(
                f"Distance {self.state['distance']:.2f}  |  "
                f"P1 whiffs {self.state['p1']['whiffs']}  |  P2 whiffs {self.state['p2']['whiffs']}"
            ),
        )
        if self.state["round_over"]:
            winner = "Draw" if self.state["winner"] == 0 else f"P{self.state['winner']} wins"
            self.canvas.create_text(
                42, distance_y + 22, text=f"ROUND OVER — {winner}", anchor="w",
                fill="#f6e05e", font=("Consolas", 11, "bold"),
            )
        if self.paused:
            self.canvas.create_text(
                self.width / 2, offset + (self.height - offset) / 2, text="PAUSED",
                fill="#f6e05e", font=("Segoe UI", 28, "bold"),
            )

    def _draw_training_status(self) -> None:
        """Live readout of what the actual training run (not this viewer) is
        doing, pulled from its own stdout log. Purely informational, and
        confined entirely to its own band so it can never overlap the fight
        display below it.
        """
        band = self.STATUS_BAND_HEIGHT
        status = self.training_status
        update = status.get("update", 0)
        stage_key = _curriculum_stage(update)[0] if update else "unknown"
        self_play = status.get("self_play")
        stage_color, stage_label = self.STAGE_ACCENTS.get(stage_key, self.STAGE_ACCENTS["unknown"])

        canvas = self.canvas
        canvas.create_rectangle(0, 0, self.width, band, fill="#0a1622", outline="")
        canvas.create_rectangle(0, band, self.width, band + 2, fill=stage_color, outline="")
        canvas.create_rectangle(10, 8, 16, band - 8, fill=stage_color, outline="")

        stage_text = stage_label
        if self_play is not None:
            stage_text = f"{stage_label}  (latest upd {self_play['latest']} / best upd {self_play['best']})"
        canvas.create_text(
            24, band / 2, anchor="w", fill="#f7fafc", font=("Segoe UI", 11, "bold"),
            text=f"TRAINING  update {update:,}",
        )
        canvas.create_text(
            190, band / 2, anchor="w", fill=stage_color, font=("Consolas", 11, "bold"),
            text=stage_text,
        )

        entropy = status.get("entropy")
        eval_win_rate = status.get("eval_win_rate")
        stochastic_win_rate = status.get("stochastic_eval_win_rate")
        if eval_win_rate is not None:
            canvas.create_text(
                self.width - 16, band / 2, anchor="e", fill="#90cdf4", font=("Consolas", 11, "bold"),
                text=f"eval win {eval_win_rate:.0%} / {stochastic_win_rate:.0%} stoch",
            )
        if entropy is not None:
            entropy_color = "#68d391"
            if entropy < 0.05:
                entropy_color = "#fc8181"
            elif entropy < 0.2:
                entropy_color = "#f6e05e"
            entropy_x = self.width - 260 if eval_win_rate is not None else self.width - 16
            canvas.create_text(
                entropy_x, band / 2, anchor="e", fill=entropy_color, font=("Consolas", 11, "bold"),
                text=f"entropy {entropy:.3f}",
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
    parser.add_argument(
        "--follow-run-dir", type=Path, default=None,
        help="Follow a full training run directory: mirrors its actual curriculum stage or "
             "self-play opponent (read from its stdout log) instead of a fixed opponent, and "
             "overlays live update/entropy/win-rate stats. Mutually exclusive with --follow-dir.",
    )
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
    if args.follow_run_dir is not None:
        args.follow_run_dir = args.follow_run_dir.resolve()
    if args.follow_dir is not None and args.follow_run_dir is not None:
        parser.error("--follow-dir and --follow-run-dir are mutually exclusive")
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

    if args.follow_run_dir is not None:
        if not args.follow_run_dir.is_dir():
            raise FileNotFoundError(f"training run directory not found: {args.follow_run_dir}")
        checkpoints_dir = args.follow_run_dir / "checkpoints"
        training_status = read_training_status(args.follow_run_dir)
        self_play = training_status.get("self_play")
        if self_play is not None:
            opponent_checkpoint = _checkpoint_for_update(checkpoints_dir, self_play["latest"])
            profile_index, profile = _resolve_profile(args, profiles)
        else:
            opponent_checkpoint = None
            profile_index, profile = _pick_stage_profile(profiles, training_status.get("update", 0))
        checkpoint = _latest_checkpoint(checkpoints_dir)
        if args.headless_steps:
            return run_headless(args, profile_index, checkpoint, opponent_checkpoint=opponent_checkpoint)
        V2Visualizer(
            args, profile_index, profile, checkpoint,
            opponent_checkpoint=opponent_checkpoint, training_status=training_status,
        ).run()
        return 0

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

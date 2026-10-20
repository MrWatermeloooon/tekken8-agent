"""Tekken 8 agent visualizer: a desktop window that animates full-move fights.

    python scripts/visualizer.py [--run runs/train_all | --checkpoint PATH]
                                 [--character jun --opponent king] [--greedy] [--scripted-opponent]

One CPU fight runs in real time (t8_v2_full_move_feed) with the checkpoint on
both sides; by default the newest champion of --run, a new random matchup after
every match (first to three rounds). Moves are sampled from the policy as in
training (--greedy: always the most likely move, which tends to stall).

The engine has positions, postures, the current move and frame, hit timing,
contacts, Heat and Rage, but no skeleton, so the fighters are posed
procedurally: each hit of a move extends the limb its notation names (1/2
punches, 3/4 kicks) toward its level, timed to its active frames; throws grab,
lift and slam; reactions follow the last hit's level; juggles arc. Effects
follow the engine's events: hit sparks and hit stop, blocks, counter hits,
Heat Burst and aura, Rage Art, wall splats, KO slow motion.

Keys: Space pause, 1/2/3 speed, N next match, G greedy/sampled, P policy, Esc quit.

    python scripts/visualizer.py --headless-frames 240   (no window; for tests)
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
ROUNDS_TO_WIN = 3
MAX_ROUNDS = 7  # a match of draws still ends
MAX_HEALTH = 180.0
FPS = 60

POSTURE = ["standing", "crouching", "airborne", "grounded", "wakeup", "wall_splat"]
UNIVERSAL = ["idle", "walk_forward", "walk_back", "dash_forward", "backdash", "crouch", "sidestep_left",
             "sidestep_right", "throw_break", "throw_break1", "throw_break2", "heat_dash", "wakeup_stand",
             "wakeup_roll", "tech_roll"]
HIGH, MID, LOW, SMID, SLOW, THROW = range(6)
# The original visualizer's player colours.
PLAYER_COLORS = ((104, 211, 145), (246, 135, 179))
MOVE_FLAGS = ("heat_burst", "heat_smash", "heat_engager", "rage_art", "power_crush", "tornado", "homing")

# top, bottom, skin, hair, gloves/accent, build
LOOKS = {
    "jin": ("#26262e", "#1d1d25", "#d9a982", "#141414", "#c8102e", 1.0),
    "kazuya": ("#4a2370", "#3b1d5a", "#d3a07a", "#181818", "#c9a24a", 1.05),
    "king": ("#d9a520", "#1c1c1c", "#c58b5c", "#d9a520", "#2a6fdb", 1.18),
    "jun": ("#f2efe6", "#ece8dc", "#efc8a6", "#1a1a1a", "#9c1f33", 0.93),
    "paul": ("#c8261c", "#c8261c", "#e3b089", "#f1d36b", "#202020", 1.05),
    "law": ("#e3b089", "#e8c032", "#e0b08a", "#1a1a1a", "#202020", 0.98),
    "jack-8": ("#69757c", "#4b555b", "#69757c", "#3a3f42", "#e0a020", 1.35),
    "lars": ("#2d3b45", "#22303a", "#e6bb95", "#d8c27a", "#58b0d8", 1.02),
    "xiaoyu": ("#e8579a", "#f2d4e3", "#f2cdb0", "#1a1a1a", "#ffd23f", 0.88),
    "nina": ("#3a2b5c", "#2b2046", "#f1cbb0", "#e9cf7e", "#b03060", 0.94),
    "leroy": ("#7a5a3a", "#2a2a2a", "#6b4a32", "#d8d8d8", "#c8a050", 1.0),
    "asuka": ("#2e5aa8", "#e8e8e8", "#efc6a6", "#3a2416", "#d03030", 0.93),
    "lili": ("#f6f4ef", "#f6f4ef", "#f6d5bf", "#f2dd8a", "#d4af37", 0.9),
    "bryan": ("#4b5320", "#3c4219", "#d6a985", "#2a2a2a", "#8a2a2a", 1.06),
    "hwoarang": ("#1f3a93", "#e9e9e9", "#e3b48e", "#b5321e", "#1f1f1f", 1.0),
    "claudio": ("#e6e3da", "#2a2f3a", "#e8bc98", "#d8d2c0", "#5aa0ff", 1.0),
    "azucena": ("#d9533a", "#2b2b2b", "#cf9a74", "#1a1a1a", "#f0c040", 0.94),
    "raven": ("#1d1d22", "#1d1d22", "#5a3a28", "#e6e6e6", "#9050d0", 1.05),
    "leo": ("#2c5a3a", "#e8e4da", "#f0c8a8", "#c89048", "#e8a020", 0.95),
    "steve": ("#e3b48e", "#24324a", "#efc4a2", "#e8d070", "#e04040", 1.02),
    "kuma": ("#6b4423", "#6b4423", "#6b4423", "#5a381c", "#2a2a2a", 1.4),
    "yoshimitsu": ("#2e7d32", "#1f4f22", "#9a9a9a", "#c0c0c0", "#d4d4d4", 1.0),
    "shaheen": ("#a8936a", "#5a4a33", "#c8916a", "#1a1a1a", "#3a6a8a", 1.02),
    "dragunov": ("#e3e3e3", "#d6d6d6", "#e9d2c2", "#2a2a2a", "#3a3a3a", 1.04),
    "feng": ("#2a2a2a", "#7a1f1f", "#e6b896", "#121212", "#d4a83a", 1.0),
    "panda": ("#f2f2f2", "#141414", "#f2f2f2", "#141414", "#141414", 1.4),
    "lee": ("#8a2b2b", "#1a1a1a", "#efc6a6", "#cfcfcf", "#d0d0d0", 1.0),
    "alisa": ("#e8a8c8", "#3a3a4a", "#f4d8c8", "#ef7aa8", "#5ad0ff", 0.9),
    "zafina": ("#2a2238", "#2a2238", "#c08a64", "#121212", "#c8a050", 0.93),
    "devil-jin": ("#121216", "#121216", "#8a7a9a", "#121212", "#b03ad8", 1.03),
    "victor": ("#22283a", "#1a1f2e", "#ecc8aa", "#e0e0e0", "#3ad0ff", 1.02),
    "reina": ("#ece6f2", "#4a3a6a", "#f2d0b6", "#2a1a3a", "#9b5cff", 0.93),
    "eddy": ("#9a6a46", "#1a7a3a", "#9a6a46", "#1a1a1a", "#e03030", 1.0),
    "lidia": ("#e8e2d6", "#c8263a", "#f0cdb2", "#1a1a1a", "#c8263a", 0.94),
    "heihachi": ("#d6a582", "#2c2c2c", "#d6a582", "#e0e0e0", "#c84a1a", 1.06),
    "clive": ("#2f3b52", "#2a2e38", "#e6bc98", "#cdbf9a", "#ff7a2a", 1.04),
    "anna": ("#a8182a", "#1a1a1a", "#f2cfb8", "#3a1a1a", "#c8a050", 0.93),
    "fahkumram": ("#8a5a3a", "#2a4a8a", "#8a5a3a", "#121212", "#e8e8e8", 1.08),
    "armor-king": ("#121212", "#121212", "#2a2a2a", "#121212", "#d4af37", 1.18),
    "miary-zo": ("#f0c86a", "#3a5a9a", "#f0cdb0", "#c84a4a", "#f06aa0", 0.88),
    "kunimitsu": ("#a8263a", "#1a1a1a", "#f0cdb0", "#1a1a1a", "#d8d8d8", 0.92),
}


def hex_rgb(text: str) -> tuple[int, int, int]:
    text = text.lstrip("#")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def look_for(slug: str):
    if slug in LOOKS:
        top, bottom, skin, hair, accent, build = LOOKS[slug]
        return tuple(map(hex_rgb, (top, bottom, skin, hair, accent))) + (build,)
    hue = (sum(ord(c) * 31 ** i for i, c in enumerate(slug)) % 360) / 360
    rgb = lambda h, s, v: tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h % 1, s, v))
    return rgb(hue, 0.55, 0.55), rgb(hue + 0.08, 0.4, 0.3), (224, 180, 148), (26, 26, 26), rgb(hue + 0.5, 0.7, 0.9), 1.0


def shade(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def clamp(value, low, high):
    return max(low, min(high, value))


def ease(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def title(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-")).replace("Jack 8", "Jack-8")


# ---------------------------------------------------------------- data
def newest(directory: Path) -> Path | None:
    found = [p for p in directory.glob("update_*.t8ppo") if p.with_suffix(".t8fmstate").exists()]
    return max(found, key=lambda p: int(p.stem.split("_")[1])) if found else None


def trained_characters(checkpoint: Path) -> list[str]:
    # .t8fmstate: 8-byte magic, u32 version, u64 update, then u64-length-prefixed
    # strings: catalog, roster, learners (comma-separated).
    data = checkpoint.with_suffix(".t8fmstate").read_bytes()
    offset, strings = 20, []
    for _ in range(3):
        length = int.from_bytes(data[offset:offset + 8], "little")
        strings.append(data[offset + 8:offset + 8 + length].decode("utf-8"))
        offset += 8 + length
    return [slug for slug in strings[2].split(",") if slug]


def load_moves() -> dict[str, dict[str, dict]]:
    moves: dict[str, dict[str, dict]] = {}
    with (REPO / "data" / "generated" / "full_combat_bindings.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            entry = {"name": row["name"], "command": row["command"]}
            for flag in MOVE_FLAGS:
                if row.get(flag, "0") not in ("", "0"):
                    entry[flag] = True
            moves.setdefault(row["character"], {})[row["stable_id"]] = entry
    return moves


def feed_binary() -> Path:
    for directory in ("build-gpu/Release", "build-gpu", "build/Release", "build"):
        path = REPO / directory / f"t8_v2_full_move_feed{EXE}"
        if path.exists():
            return path
    raise SystemExit("t8_v2_full_move_feed is not built (cmake --build build-gpu --config Release)")


class Feed:
    """The feed process for one match; frames arrive on a queue from a reader thread."""

    def __init__(self, command: list[str]) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1,
                                        creationflags=flags)
        self.frames: queue.Queue = queue.Queue(maxsize=900)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith("{"):
                try:
                    self.frames.put(json.loads(line), timeout=5)
                except queue.Full:
                    pass

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


# ---------------------------------------------------------------- poses
# 2D side-view angles, radians from straight down, positive = toward the
# opponent: ua/fa upper and fore arm, th/sh thigh and shin; lean = torso tilt
# forward from vertical, head = extra head tilt. L/R are the fighter's left and
# right limbs (1 and 3 left punch/kick, 2 and 4 right).
GUARD = dict(lean=0.12, head=-0.05, uaL=0.55, faL=2.35, uaR=0.3, faR=2.65,
             thL=0.38, shL=-0.12, thR=-0.32, shR=-0.55)
POSES = {
    "guard": GUARD,
    "crouch": dict(lean=0.55, head=-0.4, uaL=0.9, faL=2.1, uaR=0.6, faR=2.3, thL=1.35, shL=-0.35, thR=1.0, shR=-0.85),
    "block": dict(lean=0.22, head=0.3, uaL=0.95, faL=2.95, uaR=0.85, faR=3.0, thL=0.38, shL=-0.2, thR=-0.3, shR=-0.6),
    "crouch_block": dict(lean=0.6, head=-0.2, uaL=1.1, faL=2.7, uaR=1.0, faR=2.8, thL=1.35, shL=-0.35, thR=1.0, shR=-0.85),
    "hit_high": dict(lean=-0.45, head=-0.7, uaL=-0.35, faL=0.3, uaR=-0.5, faR=0.1, thL=0.2, shL=-0.2, thR=-0.45, shR=-0.7),
    "hit_mid": dict(lean=0.75, head=0.4, uaL=0.35, faL=1.1, uaR=0.25, faR=1.0, thL=0.45, shL=-0.35, thR=-0.2, shR=-0.8),
    "hit_low": dict(lean=0.35, head=0.15, uaL=0.6, faL=1.3, uaR=-0.4, faR=0.3, thL=0.9, shL=-0.75, thR=-0.25, shR=-0.4),
    "limp": dict(lean=-0.2, head=-0.5, uaL=-0.8, faL=-0.5, uaR=-1.1, faR=-0.6, thL=0.5, shL=0.1, thR=0.1, shR=-0.3),
    "kneel": dict(lean=0.65, head=0.45, uaL=0.15, faL=0.3, uaR=0.05, faR=0.2, thL=1.45, shL=-1.45, thR=0.35, shR=-1.2),
    "grab": dict(lean=0.3, head=0.0, uaL=1.35, faL=1.75, uaR=1.45, faR=1.8, thL=0.5, shL=-0.15, thR=-0.35, shR=-0.5),
    "lift": dict(lean=-0.35, head=-0.4, uaL=2.95, faL=3.1, uaR=2.85, faR=3.0, thL=0.3, shL=-0.15, thR=-0.3, shR=-0.5),
    "slam": dict(lean=0.95, head=0.3, uaL=1.6, faL=1.7, uaR=1.5, faR=1.6, thL=0.9, shL=-0.6, thR=-0.35, shR=-0.5),
    "push": dict(lean=0.15, head=0.0, uaL=1.55, faL=1.6, uaR=1.5, faR=1.55, thL=0.55, shL=-0.15, thR=-0.45, shR=-0.4),
    "flex": dict(lean=-0.15, head=-0.35, uaL=-0.9, faL=1.9, uaR=-0.9, faR=1.9, thL=0.5, shL=-0.5, thR=-0.5, shR=-0.5),
    "wall": dict(lean=-0.25, head=-0.5, uaL=-1.4, faL=-1.2, uaR=-1.5, faR=-1.3, thL=0.15, shL=-0.1, thR=-0.1, shR=-0.2),
    "victory": dict(lean=-0.05, head=-0.2, uaL=0.3, faL=2.6, uaR=3.0, faR=3.1, thL=0.1, shL=0.0, thR=-0.1, shR=-0.1),
}
KEYS = list(GUARD)


def blend(a: dict, b: dict, t: float) -> dict:
    return {k: a[k] + (b[k] - a[k]) * t for k in KEYS}


def strike_pose(limb: str, level: int) -> dict:
    high, low = level == HIGH, level in (LOW, SLOW)
    if limb == "G":
        return POSES["grab"]
    if limb == "B":
        a = 1.95 if high else 1.05 if low else 1.57
        pose = dict(GUARD, lean=0.5 if low else 0.25, uaL=a, faL=a, uaR=a - 0.05, faR=a - 0.05, thL=0.6, thR=-0.45)
        return pose
    if limb in ("L", "R"):
        a = 1.95 if high else 1.0 if low else 1.57
        pose = dict(GUARD, lean=0.55 if low else 0.28, head=0.1, thL=0.62, shL=-0.2, thR=-0.45, shR=-0.45)
        if low:
            pose.update(thL=1.3, shL=-0.4, thR=0.9, shR=-0.9)
        if limb == "L":
            pose.update(uaL=a, faL=a)
        else:
            pose.update(uaR=a, faR=a, lean=pose["lean"] + 0.12)
        return pose
    a = 2.25 if high else 0.8 if low else 1.6
    pose = dict(GUARD, lean=-0.35 if high else 0.15 if low else -0.15, head=0.2 if high else 0.0,
                uaL=0.3, faL=1.9, uaR=-0.2, faR=1.6)
    if limb == "KL":
        pose.update(thL=a, shL=a + 0.05, thR=-0.12, shR=-0.2)
    else:
        pose.update(thR=a, shR=a + 0.05, thL=-0.08, shL=-0.18)
    return pose


def limbs_for(command: str, hits: list) -> list[str]:
    segments = [s.strip() for s in (command or "").split(",") if s.strip()]
    limbs = []
    for index, hit in enumerate(hits):
        if hit[0] == THROW:
            limbs.append("G")
            continue
        segment = segments[min(index, len(segments) - 1)] if segments else ""
        buttons = {c for i, c in enumerate(segment) if c in "1234" and (i == 0 or not segment[i - 1].isalpha())}
        if {"1", "2"} <= buttons:
            limbs.append("B")
        elif "4" in buttons:
            limbs.append("KR")
        elif "3" in buttons:
            limbs.append("KL")
        elif "2" in buttons:
            limbs.append("R")
        elif "1" in buttons:
            limbs.append("L")
        else:
            limbs.append("KR" if hit[0] in (LOW, SLOW) else "R")
    return limbs


# ---------------------------------------------------------------- fighter
class Fighter:
    def __init__(self, slug: str, side: int) -> None:
        self.slug, self.side = slug, side
        self.top, self.bottom, self.skin, self.hair, self.accent, self.build = look_for(slug)
        self.state: dict | None = None
        self.previous: dict | None = None
        self.x = -1.0 if side == 0 else 1.0
        self.facing = 1 if side == 0 else -1
        self.pose = dict(GUARD)
        self.lift = 0.0
        self.vy = 0.0
        self.tilt = 0.0           # whole-body rotation about the hips (+ = backward)
        self.reaction = 0.0
        self.last_level = MID
        self.walk_phase = 0.0
        self.posture_frames = 0
        self.throw_victim: tuple[int, int] | None = None
        self.throw_attacker: tuple[int, int] | None = None
        self.pushback = 0
        self.victory = False
        self.heat_glow = 0.0
        self.rage_glow = 0.0
        self.heat_max = 600
        self.striking = False
        self.strike_limb = "R"
        self.points: dict[str, tuple[float, float]] = {}
        self.trail: list[tuple[float, float]] = []

    def target_pose(self, frame_count: int, moves: dict) -> tuple[dict, float]:
        f = self.state
        posture = POSTURE[f["posture"]] if 0 <= f["posture"] < len(POSTURE) else "standing"
        universal = UNIVERSAL[f["universal"]] if 0 <= f["universal"] < len(UNIVERSAL) else "idle"
        self.striking = False
        if self.victory:
            return POSES["victory"], 0.1
        if posture == "grounded":
            return POSES["limp"], 0.25
        if posture == "airborne" or self.throw_victim:
            return POSES["limp"], 0.25
        if posture == "wall_splat":
            return POSES["wall"], 0.4
        pose, snap = dict(GUARD), 0.35
        if posture == "wakeup":
            pose = blend(POSES["kneel"], GUARD, clamp(self.posture_frames / 20, 0, 1))
        if posture == "crouching" or universal == "crouch":
            pose = dict(POSES["crouch"])
        if universal in ("walk_forward", "walk_back", "dash_forward", "backdash"):
            swing = math.sin(self.walk_phase) * (0.65 if "dash" in universal else 0.35)
            pose = dict(pose, thL=pose["thL"] + swing, thR=pose["thR"] - swing,
                        shL=pose["shL"] - max(0, -swing) * 0.8, shR=pose["shR"] - max(0, swing) * 0.8,
                        lean=pose["lean"] + (0.25 if universal == "dash_forward" else -0.15 if universal == "backdash" else 0))
        if universal in ("sidestep_left", "sidestep_right"):
            pose = dict(pose, lean=pose["lean"] + 0.1, thL=pose["thL"] + 0.2, thR=pose["thR"] - 0.1)
        if universal.startswith("throw_break") or self.pushback > 0:
            pose, snap = blend(pose, POSES["push"], 1.0 if universal.startswith("throw_break") else min(1, self.pushback / 8)), 0.5
        if f["blockstun"]:
            pose, snap = dict(POSES["crouch_block"] if posture == "crouching" else POSES["block"]), 0.55
        elif f["stun"] > 0 and not f["move_id"]:
            reaction = POSES["hit_high"] if self.last_level == HIGH else \
                POSES["hit_low"] if self.last_level in (LOW, SLOW) else POSES["hit_mid"]
            if f.get("crumpled"):
                pose = blend(pose, POSES["kneel"], clamp(f["stun"] / 30, 0.4, 1))
            else:
                pose = blend(pose, reaction, clamp(0.45 + self.reaction, 0, 1))
            snap = 0.5
        if f["move_id"]:
            info = moves.get(f["move_id"], {})
            hits, t = f.get("hits") or [], f["action_frame"]
            if self.throw_attacker:
                start, length = self.throw_attacker
                p = clamp((frame_count - start) / length, 0, 1)
                pose = blend(POSES["grab"], POSES["lift"], ease(p / 0.5)) if p < 0.5 else \
                    blend(POSES["lift"], POSES["slam"], ease((p - 0.5) / 0.5))
                snap = 0.3
            elif info.get("heat_burst"):
                pose, snap = blend(pose, POSES["flex"], clamp(t / 8, 0, 1)), 0.4
            elif hits:
                limbs = limbs_for(info.get("command", ""), hits)
                weight, which = 0.0, 0
                for i, (_, first, active, *_rest) in enumerate(hits):
                    windup = min(9, max(3, first - 1))
                    nxt = hits[i + 1][1] - min(9, max(3, hits[i + 1][1] - first - active)) if i + 1 < len(hits) \
                        else f["move_frames"]
                    retract = max(4, min(16, nxt - (first + active)))
                    w = 0.0
                    if first - windup <= t < first:
                        w = ease((t - (first - windup)) / windup)
                    elif first <= t < first + active + 2:
                        w = 1.0
                    elif t >= first + active + 2:
                        w = 1 - ease((t - first - active - 2) / retract)
                    if w > 0.001 and t >= first - windup:
                        weight, which = w, i
                strike = strike_pose(limbs[which], hits[which][0])
                chamber = blend(pose, strike, 0.25)
                pose = blend(chamber, strike, weight) if weight > 0 else blend(pose, chamber, clamp(t / 6, 0, 1))
                snap = 0.62
                first, active = hits[which][1], hits[which][2]
                self.striking = first - 1 <= t <= first + active + 1
                self.strike_limb = limbs[which]
            else:
                pose = blend(pose, POSES["flex"], 0.2)
        return pose, snap

    def update(self, frame_count: int, steps: int, moves: dict, other: "Fighter", alpha: float) -> None:
        f = self.state
        if not f:
            return
        posture = POSTURE[f["posture"]] if 0 <= f["posture"] < len(POSTURE) else "standing"
        self.x += (f["x"] - self.x) * clamp(0.5 * max(steps, alpha) + 0.2, 0, 1)
        face = 1 if (other.state and other.state["x"] >= f["x"]) else -1
        if f.get("back_turned"):
            face = -face
        if steps:
            self.facing = face
        for key in ("throw_victim", "throw_attacker"):
            value = getattr(self, key)
            if value and frame_count - value[0] > value[1]:
                setattr(self, key, None)
        for _ in range(steps):
            if posture == "airborne":
                self.vy -= 0.0042
                self.lift = max(0.25, self.lift + self.vy)
                self.tilt += (1.25 + max(0, self.vy) * 3 - self.tilt) * 0.12
            else:
                self.vy = 0
                self.lift = max(0.0, self.lift - 0.08)
                target = math.pi / 2 if posture == "grounded" else 0.0
                self.tilt += (target - self.tilt) * 0.2
            self.reaction = max(0.0, self.reaction - 0.05)
            self.pushback = max(0, self.pushback - 1)
            universal = UNIVERSAL[f["universal"]] if 0 <= f["universal"] < len(UNIVERSAL) else "idle"
            if universal in ("walk_forward", "walk_back"):
                self.walk_phase += 0.19
            elif universal in ("dash_forward", "backdash"):
                self.walk_phase += 0.32
        target, snap = self.target_pose(frame_count, moves)
        k = 1 - (1 - snap) ** alpha if alpha > 0 else 0
        for key in KEYS:
            self.pose[key] += (target[key] - self.pose[key]) * k
        self.heat_glow += ((1.0 if f["heat"] else 0.0) - self.heat_glow) * 0.1
        self.rage_glow += ((1.0 if f["rage"] else 0.0) - self.rage_glow) * 0.1
        self.solve()

    def solve(self) -> None:
        """Forward kinematics in metres (x toward the opponent, y up), then facing, tilt and floor."""
        p, b = self.pose, self.build
        height = 1 + (b - 1) * 0.35
        thigh, shin, upper, fore = 0.44 * height, 0.43 * height, 0.29 * height, 0.27 * height
        torso, neck = 0.52 * height, 0.1
        hips = (0.0, 0.95 * height)
        down = lambda a, length, origin: (origin[0] + math.sin(a) * length, origin[1] - math.cos(a) * length)
        up = lambda a, length, origin: (origin[0] + math.sin(a) * length, origin[1] + math.cos(a) * length)
        chest = up(p["lean"], torso, hips)
        shoulder = up(p["lean"], torso - 0.06, hips)
        head = up(p["lean"] + p["head"], neck + 0.12, chest)
        points = {"hips": hips, "chest": chest, "head": head, "shoulder": shoulder}
        for side in "LR":
            elbow = down(p["ua" + side], upper, shoulder)
            points["elbow" + side], points["fist" + side] = elbow, down(p["fa" + side], fore, elbow)
            knee = down(p["th" + side], thigh, hips)
            points["knee" + side], points["ankle" + side] = knee, down(p["sh" + side], shin, knee)
        # Whole-body tilt about the hips (juggles, knockdowns, throws).
        tilt, lift, slide = self.tilt, self.lift, 0.0
        if self.throw_victim:
            start, length = self.throw_victim
            q = clamp((self.frame_count - start) / length, 0, 1)
            tilt, lift, slide = q * math.pi * 0.9, math.sin(math.pi * min(1, q / 0.85)) * 1.1, -math.sin(q * math.pi) * 0.4
        if abs(tilt) > 1e-3:
            c, s = math.cos(tilt), math.sin(tilt)
            hx, hy = hips
            for key, (x, y) in points.items():
                dx, dy = x - hx, y - hy
                # Counter-clockwise in the local frame: positive tilt falls backward.
                points[key] = (hx + dx * c - dy * s, hy + dx * s + dy * c)
        lowest = min(y for _, y in points.values())
        upright = self.tilt < 0.3 and not self.throw_victim and (not self.state or self.state["posture"] != 2)
        if upright:
            offset = -min(points["ankleL"][1], points["ankleR"][1]) + 0.04
        elif self.state and POSTURE[self.state["posture"]] == "grounded" and abs(tilt) > 1:
            offset = -lowest + 0.12
        else:
            offset = max(-lowest + 0.05, 0.0) + lift
        self.points = {key: (self.x + self.facing * (x + slide), y + offset) for key, (x, y) in points.items()}
        self.radius = 0.11 * b

    frame_count = 0

    def tip(self, limb: str) -> tuple[float, float]:
        if limb in ("L", "B", "G"):
            return self.points.get("fistL", (self.x, 1))
        if limb == "R":
            return self.points.get("fistR", (self.x, 1))
        return self.points.get("ankleL" if limb == "KL" else "ankleR", (self.x, 0.5))


# ---------------------------------------------------------------- the app
class App:
    def __init__(self, args: argparse.Namespace, headless: bool = False) -> None:
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        import pygame
        import pygame.gfxdraw
        self.pg, self.gfx = pygame, pygame.gfxdraw
        pygame.init()
        self.args, self.headless = args, headless
        flags = 0 if headless else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((1280, 720), flags)
        pygame.display.set_caption("Tekken 8 Agent - Visualizer")
        self.clock = pygame.time.Clock()
        font = lambda size, bold=True: pygame.font.SysFont("bahnschrift,segoeui,arial", size, bold=bold)
        self.fonts = {k: font(v) for k, v in dict(name=30, timer=50, small=17, tiny=14, caption=26, cmd=20,
                                                  combo=58, callout=36, banner=128, banner_small=84).items()}
        self.moves = load_moves()
        self.random = random.Random(args.seed)
        self.greedy = args.greedy
        self.feed: Feed | None = None
        self.match = {}
        self.fighters: list[Fighter] = []
        self.buffer: list[dict] = []
        self.current: dict | None = None
        self.frame_count = 0
        self.paused, self.speed = False, 1.0
        self.slowmo, self.slowmo_frames, self.hitstop, self.shake = 1.0, 0, 0.0, 0.0
        self.accumulator = 0.0
        self.cinematic: dict | None = None
        self.wins = [0, 0]
        self.round_index, self.round_over, self.match_over_at = -1, False, None
        self.show_policy = False
        self.particles: list[list] = []   # x, y, vx, vy, life, max, size, color, gravity
        self.flashes: list[list] = []     # x, y, life, max, size, color, kind
        self.texts: list[list] = []       # text, side, life, max, color
        self.banner: list | None = None   # text, life, max, color, small
        self.captions = [None, None]      # [name, command, flags, life]
        self.combo = [[0, 0.0, 0], [0, 0.0, 0]]  # hits, damage, life
        self.lag = [MAX_HEALTH, MAX_HEALTH]
        self.cam = [0.0, 1.0, 230.0]      # x, y, pixels per metre
        self.glows: dict = {}
        self.background = None
        self.shake_offset = (0.0, 0.0)
        self.start_match()

    # ---------------------------------------------------------- match control
    def checkpoint(self) -> Path | None:
        if self.args.checkpoint:
            return self.args.checkpoint
        return newest(self.args.run / "champion") or newest(self.args.run / "checkpoints")

    def start_match(self) -> None:
        if self.feed:
            self.feed.close()
        checkpoint = self.checkpoint()
        roster = trained_characters(checkpoint) if checkpoint else sorted(self.moves)
        p1 = self.args.character or self.random.choice(roster)
        p2 = self.args.opponent or self.random.choice(roster)
        command = [str(feed_binary()), "--character", p1, "--opponent", p2, "--interval-ms", "15",
                   "--seed", str(self.random.randrange(1 << 30)), "--allow-provisional"]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
            if not self.args.scripted_opponent:
                command += ["--opponent-checkpoint", str(checkpoint)]
        if not self.greedy:
            command.append("--sample")
        if self.headless:
            command[command.index("--interval-ms") + 1] = "0"
        self.feed = Feed(command)
        self.match = {"p1": p1, "p2": p2, "checkpoint": checkpoint,
                      "update": int(checkpoint.stem.split("_")[1]) if checkpoint else None,
                      "champion": bool(checkpoint and checkpoint.parent.name == "champion")}
        self.fighters = [Fighter(p1, 0), Fighter(p2, 1)]
        self.buffer.clear()
        self.current = None
        self.wins = [0, 0]
        self.round_index, self.round_over, self.match_over_at = -1, False, None
        self.rounds_played = 0
        self.cinematic, self.slowmo_frames, self.hitstop = None, 0, 0
        self.particles.clear(); self.flashes.clear(); self.texts.clear()
        self.captions = [None, None]
        pygame = self.pg
        pygame.display.set_caption(f"Tekken 8 Agent - {title(p1)} vs {title(p2)}")

    # ---------------------------------------------------------- effects
    def glow(self, radius: int, color, intensity: float = 1.0) -> "pygame.Surface":
        """A soft additive light: the colour fading to black (additive blending ignores alpha)."""
        level = max(1, min(8, round(intensity * 8)))
        key = (radius, color, level)
        if key not in self.glows:
            pg = self.pg
            surface = pg.Surface((radius * 2, radius * 2))
            surface.fill((0, 0, 0))
            for r in range(radius, 0, -1):
                t = (1 - r / radius) ** 1.8 * level / 8
                pg.draw.circle(surface, tuple(int(c * t) for c in color), (radius, radius), r)
            self.glows[key] = surface
            if len(self.glows) > 400:
                self.glows.pop(next(iter(self.glows)))
        return self.glows[key]

    def burst(self, x, y, color, count, speed, life, size, gravity=-0.004):
        for _ in range(count):
            angle = self.random.uniform(0, math.tau)
            v = speed * self.random.uniform(0.3, 1.2)
            self.particles.append([x, y, math.cos(angle) * v, math.sin(angle) * v + speed * 0.3,
                                   life * self.random.uniform(0.6, 1.2), life, size * self.random.uniform(0.6, 1.4),
                                   color, gravity])
        if len(self.particles) > 1400:
            del self.particles[:len(self.particles) - 1400]

    def flash(self, x, y, color, size, life, kind="star"):
        self.flashes.append([x, y, life, life, size, color, kind])

    def callout(self, side, text, color):
        self.texts.append([text, side, 70, 70, color])

    def show_banner(self, text, color=(255, 215, 94), frames=70, small=False):
        self.banner = [text, frames, frames, color, small]

    # ---------------------------------------------------------- frames
    def apply_frame(self, frame: dict) -> None:
        previous, self.current = self.current, frame
        self.frame_count += 1
        Fighter.frame_count = self.frame_count
        if frame["round"] != self.round_index:
            # The feed starts the next round on a KO or when the timer runs out; a timeout
            # never sets a winner, so score it here like Tekken: more health wins, equal
            # health is a draw and both get the round.
            if previous and self.round_index >= 0 and not self.round_over:
                health = [f["health"] for f in previous["fighters"]]
                self.end_round(0 if health[0] > health[1] else 1 if health[1] > health[0] else -1, timeout=True)
            if self.match_over_at is None and self.round_index >= 0 and self.newer_checkpoint():
                self.start_match()  # a new champion: show it from the next round
                return
            self.round_index, self.round_over = frame["round"], False
            for fighter in self.fighters:
                fighter.victory, fighter.lift, fighter.vy, fighter.tilt = False, 0.0, 0.0, 0.0
                fighter.throw_victim = fighter.throw_attacker = None
            n = self.rounds_played + 1
            final = max(self.wins) == ROUNDS_TO_WIN - 1 and self.wins[0] == self.wins[1]
            self.show_banner("FINAL ROUND" if final else f"ROUND {n}", small=True, frames=60)
            self.pending_fight = 62
            self.lag = [MAX_HEALTH, MAX_HEALTH]
        for side, f in enumerate(frame["fighters"]):
            fighter = self.fighters[side]
            f["move_id"] = f["move"]
            fighter.previous, fighter.state = fighter.state, f
            same = fighter.previous and fighter.previous["posture"] == f["posture"]
            fighter.posture_frames = fighter.posture_frames + 1 if same else 0
        for side, fighter in enumerate(self.fighters):
            f, p = fighter.state, fighter.previous
            if not p:
                continue
            table = self.moves.get(fighter.slug, {})
            chest = fighter.points.get("chest", (fighter.x, 1.2))
            if f["move_id"] and (f["move_id"] != p["move_id"] or f["action_frame"] < p["action_frame"]):
                info = table.get(f["move_id"], {})
                flags = [label for key, label in (("rage_art", "RAGE ART"), ("heat_burst", "HEAT BURST"),
                                                  ("heat_smash", "HEAT SMASH"), ("heat_engager", "HEAT ENGAGER"),
                                                  ("power_crush", "POWER CRUSH"), ("tornado", "TORNADO"),
                                                  ("homing", "HOMING")) if info.get(key)]
                self.captions[side] = [info.get("name", f["move_id"]), info.get("command", ""), " · ".join(flags), 90]
                if info.get("rage_art"):
                    self.cinematic = {"kind": "rage", "fighter": side, "until": self.frame_count + 70}
                    self.slowmo, self.slowmo_frames = 0.35, 50
                    self.show_banner("RAGE ART", (255, 70, 50), 60, small=True)
                    self.burst(*chest, (255, 50, 30), 90, 0.07, 40, 0.06, 0.0005)
                if info.get("heat_smash"):
                    self.callout(side, "HEAT SMASH", (80, 214, 255))
                    self.cinematic = {"kind": "smash", "fighter": side, "until": self.frame_count + 40}
                    self.burst(*chest, (80, 214, 255), 70, 0.07, 34, 0.05, 0.0005)
            if f["heat"] and not p["heat"]:
                info = table.get(f["move_id"], {})
                self.show_banner("HEAT BURST" if info.get("heat_burst") else "HEAT", (90, 210, 255), 50, small=True)
                self.flash(*chest, (190, 242, 255), 2.2, 22, "ring")
                self.flash(fighter.x, 0.05, (80, 214, 255), 3.0, 30, "floor")
                self.burst(*chest, (127, 228, 255), 90, 0.1, 34, 0.06, 0.0004)
                self.hitstop, self.shake = max(self.hitstop, 10), max(self.shake, 10)
                self.screen_flash = [(191, 242, 255), 0.45, 18, 18]
                fighter.heat_max = max(fighter.heat_max, f["heat_frames"])
            if f["rage"] and not p["rage"]:
                self.callout(side, "RAGE", (255, 50, 50))
                self.burst(*chest, (255, 50, 30), 50, 0.05, 30, 0.05, 0.0006)
            if f.get("heat_dash") and not p.get("heat_dash"):
                self.callout(side, "HEAT DASH", (80, 214, 255))
            now, before = POSTURE[f["posture"]], POSTURE[p["posture"]]
            if now == "airborne" and before != "airborne":
                fighter.vy, fighter.lift = 0.085, max(fighter.lift, 0.3)
            if now == "grounded" and before != "grounded":
                self.burst(fighter.x, 0.05, (200, 184, 160), 26, 0.035, 30, 0.07, -0.001)
                self.shake = max(self.shake, 5)
            if now == "wall_splat" and before != "wall_splat":
                wall_x = frame["stage"]["left"] if fighter.x < 0 else frame["stage"]["right"]
                self.flash(wall_x, 1.2, (255, 224, 160), 2.0, 16)
                self.burst(wall_x, 1.2, (255, 208, 122), 40, 0.06, 26, 0.05)
                self.callout(1 - side, "WALL SPLAT", (255, 215, 94))
                self.shake, self.hitstop = max(self.shake, 9), max(self.hitstop, 6)
            moving = UNIVERSAL[f["universal"]] if 0 <= f["universal"] < len(UNIVERSAL) else "idle"
            if moving in ("dash_forward", "backdash") and f["universal"] != p["universal"]:
                self.burst(fighter.x, 0.04, (150, 138, 120), 10, 0.02, 22, 0.05, 0)
        if previous and frame["stage"]["level"] > previous["stage"]["level"]:
            mid = sum(f.x for f in self.fighters) / 2
            self.show_banner("FLOOR BREAK", small=True, frames=50)
            self.burst(mid, 0.05, (216, 200, 168), 120, 0.08, 45, 0.08, -0.002)
            self.shake = max(self.shake, 14)
        if frame["contact_age"] == 0:
            for contact in frame["contacts"]:
                self.contact(contact)
        if frame["winner"] >= 0 and not self.round_over:
            self.end_round(frame["winner"] if frame["winner"] in (0, 1) else -1, timeout=False)

    def end_round(self, winner: int, timeout: bool) -> None:
        """Score a round (winner -1: a draw, both get it) and end the match when it is decided."""
        self.round_over = True
        self.rounds_played += 1
        if winner in (0, 1):
            self.wins[winner] += 1
            loser = self.fighters[1 - winner]
            ko = not timeout and loser.state["health"] <= 0
            self.show_banner("K.O." if ko else "TIME", (255, 90, 70), 90)
            self.screen_flash = [(255, 255, 255), 0.6, 30, 30]
            self.slowmo, self.slowmo_frames = 0.25, 70
            self.cinematic = {"kind": "ko", "fighter": 1 - winner, "until": self.frame_count + 110}
            self.fighters[winner].victory = True
        else:
            self.wins = [self.wins[0] + 1, self.wins[1] + 1]
            self.show_banner("DRAW", small=True)
        decided = max(self.wins) >= ROUNDS_TO_WIN or self.rounds_played >= MAX_ROUNDS
        if decided and self.match_over_at is None:
            self.match_over_at = self.frame_count
            if self.wins[0] == self.wins[1]:
                text = "DRAW GAME"
            else:
                text = f"{title(self.fighters[0 if self.wins[0] > self.wins[1] else 1].slug).upper()} WINS"
            self.winner_banner = (self.frame_count + 100, text)

    def newer_checkpoint(self) -> bool:
        if self.args.checkpoint or not self.match.get("checkpoint"):
            return False
        latest = self.checkpoint()
        return latest is not None and latest != self.match["checkpoint"]

    def contact(self, contact: dict) -> None:
        attacker, defender = self.fighters[contact["attacker"]], self.fighters[1 - contact["attacker"]]
        level = contact["level"] if contact.get("level", -1) >= 0 else MID
        kind, damage = contact["contact"], contact["damage"]
        height = 1.55 if level == HIGH else 0.4 if level in (LOW, SLOW) else 1.05
        at = attacker.tip(attacker.strike_limb) if attacker.striking else \
            (defender.x + (attacker.x - defender.x) * 0.35, height + defender.lift)
        big = clamp(damage / 25, 0.3, 1.6)
        if kind in ("hit", "counter_hit") and level == THROW:
            self.burst(defender.x, 0.08, (216, 200, 168), 50, 0.06, 36, 0.07, -0.002)
            self.flash(defender.x, 0.3, (255, 241, 192), 2.4, 16)
            self.flash(defender.x, 0.05, (255, 224, 160), 2.6, 24, "floor")
            self.shake, self.hitstop = max(self.shake, 10 + big * 6), max(self.hitstop, 8)
            self.add_combo(contact["attacker"], defender, damage)
            return
        if kind in ("hit", "counter_hit"):
            counter = kind == "counter_hit"
            defender.last_level, defender.reaction = level, 0.6 if counter else 0.35
            self.flash(*at, (255, 122, 42) if counter else (255, 241, 192), (1.4 if counter else 1.0) * (0.7 + big * 0.6),
                       16 if counter else 11)
            self.burst(*at, (255, 106, 32) if counter else (255, 210, 122), int(14 + big * 22), 0.06 + big * 0.02,
                       22, 0.035)
            if attacker.state["heat"]:
                self.burst(*at, (111, 224, 255), 20, 0.07, 22, 0.04)
            self.hitstop = max(self.hitstop, clamp(3 + damage / 5, 3, 12 if counter else 9))
            self.shake = max(self.shake, 3 + big * 7 + (5 if counter else 0))
            if counter:
                self.callout(contact["attacker"], "COUNTER HIT", (255, 138, 58))
                self.slowmo, self.slowmo_frames = 0.5, 14
            self.add_combo(contact["attacker"], defender, damage)
        elif kind == "blocked":
            x = defender.x + (attacker.x - defender.x) * 0.25
            self.flash(x, height, (90, 184, 255), 1.0, 12, "shield")
            self.burst(x, height, (138, 208, 255), 8, 0.035, 14, 0.03, 0)
            self.hitstop, self.shake = max(self.hitstop, 3), max(self.shake, 2)
        elif kind in ("parried", "reversed"):
            self.flash(*at, (255, 255, 255), 2.0, 16)
            self.flash(*at, (255, 255, 255), 1.6, 18, "ring")
            self.callout(1 - contact["attacker"], "PARRY" if kind == "parried" else "REVERSAL", (255, 255, 255))
            self.hitstop = max(self.hitstop, 10)
        elif kind == "armored":
            chest = defender.points.get("chest", (defender.x, 1.2))
            self.flash(*chest, (176, 112, 255), 1.6, 14)
            self.burst(*chest, (176, 112, 255), 24, 0.05, 20, 0.04, 0)
            self.callout(1 - contact["attacker"], "POWER CRUSH", (194, 138, 255))
            self.hitstop = max(self.hitstop, 6)
        elif kind == "throw_held":
            length = max(30, (attacker.state.get("move_frames") or 60) - attacker.state["action_frame"])
            attacker.throw_attacker = (self.frame_count, length)
            defender.throw_victim = (self.frame_count, length)
        elif kind == "throw_broken":
            mid = ((attacker.x + defender.x) / 2, 1.2)
            self.flash(*mid, (255, 255, 255), 2.2, 16)
            self.burst(*mid, (255, 255, 255), 36, 0.06, 22, 0.04, 0)
            self.callout(1 - contact["attacker"], "THROW BREAK", (255, 255, 255))
            attacker.pushback = defender.pushback = 14
            self.hitstop = max(self.hitstop, 8)

    def add_combo(self, side: int, defender: Fighter, damage: float) -> None:
        hits = defender.state.get("combo_hits", 0)
        combo = self.combo[side]
        combo[1] = combo[1] + damage if hits >= 2 else damage
        combo[0], combo[2] = hits, 90 if hits >= 2 else combo[2]

    # ---------------------------------------------------------- loop
    def step(self, dt: float) -> None:
        # Pull frames from the feed; play at speed, with hit stop, slow motion and catch-up.
        if self.feed:
            try:
                while True:
                    self.buffer.append(self.feed.frames.get_nowait())
            except queue.Empty:
                pass
        if len(self.buffer) > 240 and not self.cinematic:
            del self.buffer[:len(self.buffer) - 90]
        steps = 0
        if not self.paused:
            catch_up = 1.6 if len(self.buffer) > 40 else 1.2 if len(self.buffer) > 20 else 1.0
            speed = self.speed * (self.slowmo if self.slowmo_frames > 0 else catch_up)
            if self.hitstop > 0:
                self.hitstop -= dt * 60
            else:
                self.accumulator += dt * 60 * speed
                while self.accumulator >= 1 and self.buffer:
                    self.apply_frame(self.buffer.pop(0))
                    self.accumulator -= 1
                    steps += 1
                    self.slowmo_frames = max(0, self.slowmo_frames - 1)
                    if getattr(self, "pending_fight", 0) > 0:
                        self.pending_fight -= 1
                        if self.pending_fight == 0:
                            self.show_banner("FIGHT!", frames=40)
                    if self.hitstop > 0:
                        self.accumulator = 0
                        break
                if not self.buffer:
                    self.accumulator = min(self.accumulator, 1)
        frozen = self.paused or self.hitstop > 0
        alpha = 0.0 if frozen else dt * 60 * self.speed * (self.slowmo if self.slowmo_frames > 0 else 1)
        if self.current:
            for i, fighter in enumerate(self.fighters):
                fighter.update(self.frame_count, steps, self.moves.get(fighter.slug, {}), self.fighters[1 - i], alpha)
                trail_point = fighter.tip(fighter.strike_limb)
                if fighter.striking and steps:
                    fighter.trail.insert(0, trail_point)
                elif fighter.trail:
                    fighter.trail.pop()
                del fighter.trail[10:]
        fx = 0.0 if self.paused else dt * 60 * self.speed * (self.slowmo if self.slowmo_frames > 0 else 1)
        self.update_effects(fx)
        if self.match_over_at and self.frame_count - self.match_over_at > 220:
            self.start_match()
        banner = getattr(self, "winner_banner", None)
        if banner and self.frame_count >= banner[0]:
            self.show_banner(banner[1], frames=150, small=True)
            self.winner_banner = None

    def update_effects(self, fx: float) -> None:
        for p in self.particles:
            p[3] += p[8] * fx
            damping = 0.96 ** fx
            p[2] *= damping
            p[3] *= damping
            p[0] += p[2] * fx
            p[1] = max(0.01, p[1] + p[3] * fx)
            p[4] -= fx
        self.particles = [p for p in self.particles if p[4] > 0]
        for f in self.flashes:
            f[2] -= fx
        self.flashes = [f for f in self.flashes if f[2] > 0]
        for t in self.texts:
            t[2] -= fx
        self.texts = [t for t in self.texts if t[2] > 0]
        if self.banner:
            self.banner[1] -= fx
            if self.banner[1] <= 0:
                self.banner = None
        for caption in self.captions:
            if caption:
                caption[3] -= fx
        for combo in self.combo:
            combo[2] = max(0, combo[2] - fx)
        self.shake *= 0.86 ** fx
        flash = getattr(self, "screen_flash", None)
        if flash:
            flash[2] -= fx
            if flash[2] <= 0:
                self.screen_flash = None
        # Heat and Rage auras.
        for fighter in self.fighters:
            f = fighter.state
            if f and (f["heat"] or f["rage"]) and fx > 0 and fighter.points:
                for _ in range(int(fx * (2 if f["heat"] else 1) + self.random.random())):
                    key = self.random.choice(["chest", "elbowL", "elbowR", "kneeL", "kneeR", "head", "hips"])
                    x, y = fighter.points[key]
                    color = (63, 200, 255) if f["heat"] else (255, 42, 20)
                    if self.random.random() < 0.35:
                        color = (232, 251, 255) if f["heat"] else (255, 154, 58)
                    self.particles.append([x + self.random.uniform(-0.1, 0.1), y, self.random.uniform(-0.003, 0.003),
                                           0.012 + self.random.random() * 0.012, 24, 24,
                                           0.03 + self.random.random() * 0.03, color, 0.0002])

    # ---------------------------------------------------------- drawing
    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.screen.get_size()
        ox, oy = self.shake_offset
        return (int(w / 2 + (x - self.cam[0]) * self.cam[2] + ox), int(h * 0.80 - (y - self.cam[1] + 1.0) * self.cam[2] + oy))

    def update_camera(self, dt: float) -> None:
        if not self.fighters:
            return
        a, b = self.fighters
        w, _ = self.screen.get_size()
        mid = (a.x + b.x) / 2
        separation = abs(a.x - b.x)
        lift = max(a.lift, b.lift)
        target_x, target_y = mid, 1.0 + lift * 0.25
        target_scale = clamp(w * 0.5 / (separation + 4.6), w * 0.075, w * 0.13)
        follow = 0.08
        if self.cinematic and self.frame_count < self.cinematic["until"]:
            f = self.fighters[self.cinematic["fighter"]]
            kind = self.cinematic["kind"]
            target_x = f.x
            target_scale *= 1.55 if kind == "rage" else 1.3 if kind == "ko" else 1.2
            target_y = 1.0 + (f.lift * 0.5 if kind == "ko" else 0.1)
            follow = 0.1
        else:
            self.cinematic = None
        k = 1 - (1 - follow) ** (dt * 60)
        self.cam[0] += (target_x - self.cam[0]) * k
        self.cam[1] += (target_y - self.cam[1]) * k
        self.cam[2] += (target_scale - self.cam[2]) * k

    def draw_background(self) -> None:
        pg = self.pg
        w, h = self.screen.get_size()
        if not self.background or self.background.get_size() != (w, h):
            surface = pg.Surface((w, h))
            for y in range(h):
                t = y / h
                color = mix((10, 14, 38), (58, 29, 58), clamp(t / 0.55, 0, 1)) if t < 0.55 else \
                    mix((58, 29, 58), (150, 66, 40), clamp((t - 0.55) / 0.25, 0, 1))
                pg.draw.line(surface, color, (0, y), (w, y))
            rng = random.Random(7)
            for _ in range(120):
                surface.set_at((rng.randrange(w), rng.randrange(int(h * 0.45))), (200, 200, 230))
            surface.blit(self.glow(90, (255, 220, 170)), (int(w * 0.78) - 90, int(h * 0.12) - 90),
                         special_flags=pg.BLEND_ADD)
            pg.draw.circle(surface, (255, 236, 205), (int(w * 0.78), int(h * 0.12)), 26)
            self.background = surface
        self.screen.blit(self.background, (0, 0))
        horizon = self.to_screen(0, 0)[1] - int(self.cam[2] * 1.6)
        # Mountains and temple gates with parallax.
        for layer, (parallax, color, peaks) in enumerate(((0.15, (22, 16, 34), 9), (0.3, (14, 10, 22), 7))):
            offset = -self.cam[0] * self.cam[2] * parallax
            points = [(0, h)]
            rng = random.Random(layer + 3)
            span = w * 1.6
            for i in range(peaks * 2 + 1):
                x = (i / (peaks * 2)) * span - w * 0.3 + offset % (span / peaks) - span / peaks / 2
                y = horizon - (rng.uniform(60, 170) * (1.4 - layer * 0.4) if i % 2 else rng.uniform(0, 40))
                points.append((x, y))
            points.append((w, h))
            pg.draw.polygon(self.screen, color, points)
        for gx in (-8.5, 8.5):  # torii gates behind the stage
            sx, sy = self.to_screen(self.cam[0] + (gx - self.cam[0]) * 0.6, 0)
            scale = self.cam[2] * 0.6
            red, dark = (110, 22, 18), (34, 20, 18)
            pg.draw.rect(self.screen, red, (sx - 1.8 * scale, horizon - 4.5 * scale, 0.3 * scale, 4.5 * scale))
            pg.draw.rect(self.screen, red, (sx + 1.5 * scale, horizon - 4.5 * scale, 0.3 * scale, 4.5 * scale))
            pg.draw.rect(self.screen, red, (sx - 2.4 * scale, horizon - 4.1 * scale, 4.8 * scale, 0.3 * scale))
            pg.draw.rect(self.screen, dark, (sx - 2.8 * scale, horizon - 4.7 * scale, 5.6 * scale, 0.35 * scale))
        for i, lx in enumerate((-6, -2, 2, 6)):  # lanterns
            sx, _ = self.to_screen(self.cam[0] + (lx - self.cam[0]) * 0.7, 0)
            sy = horizon - int(self.cam[2] * (2.6 + (i % 2) * 0.3))
            flicker = 0.85 + 0.15 * math.sin(self.frame_count * 0.13 + i * 1.7)
            r = int(self.cam[2] * 0.5 * flicker)
            self.screen.blit(self.glow(max(4, r), (255, 150, 60)), (sx - r, sy - r), special_flags=pg.BLEND_ADD)
            pg.draw.circle(self.screen, (255, 170, 80), (sx, sy), max(2, int(self.cam[2] * 0.09)))
        # Floor: perspective tiles that move with the camera.
        floor_y = self.to_screen(0, 0)[1]
        floor = pg.Rect(0, horizon, w, h - horizon)
        pg.draw.rect(self.screen, (34, 30, 28), floor)
        for row in range(14):
            t = row / 14
            y = horizon + (floor_y - horizon) * (t ** 0.6) if row else horizon
            pg.draw.line(self.screen, (52, 46, 42), (0, int(y)), (w, int(y)))
        for i in range(-30, 31):
            wx = math.floor(self.cam[0]) + i * 0.75
            bx, _ = self.to_screen(wx, 0)
            tx = w / 2 + (bx - w / 2) * 0.35
            pg.draw.aaline(self.screen, (52, 46, 42), (tx, horizon), (bx + (bx - w / 2) * 0.9, h))
        pg.draw.line(self.screen, (120, 100, 70), (0, horizon), (w, horizon), 2)
        # Emblem under the fight.
        ex, ey = self.to_screen(0, 0)
        self.gfx.aaellipse(self.screen, ex, ey, int(self.cam[2] * 1.9), int(self.cam[2] * 0.22), (150, 120, 60))
        self.gfx.aaellipse(self.screen, ex, ey, int(self.cam[2] * 1.5), int(self.cam[2] * 0.17), (120, 96, 50))
        # Walls at the stage edges.
        if self.current:
            for edge, sign in ((self.current["stage"]["left"], -1), (self.current["stage"]["right"], 1)):
                x0, _ = self.to_screen(edge + sign * 0.22, 0)
                width = int(self.cam[2] * 0.9)
                top = floor_y - int(self.cam[2] * 3.2)
                rect = pg.Rect(x0 if sign > 0 else x0 - width, top, width, floor_y - top + int(self.cam[2] * 0.08))
                pg.draw.rect(self.screen, (44, 38, 34), rect)
                brick = max(6, int(self.cam[2] * 0.3))
                for row, y in enumerate(range(top, floor_y, brick)):
                    pg.draw.line(self.screen, (30, 26, 23), (rect.left, y), (rect.right, y))
                    for x in range(rect.left + (row % 2) * brick, rect.right, brick * 2):
                        pg.draw.line(self.screen, (30, 26, 23), (x, y), (x, y + brick))
                trim_x = rect.left if sign > 0 else rect.right
                pg.draw.line(self.screen, (255, 179, 71), (trim_x, top), (trim_x, floor_y), 3)

    def limb(self, a, b, width, color) -> None:
        """A thick line with round caps (the original visualizer's style), antialiased."""
        gfx = self.gfx
        (ax, ay), (bx, by) = a, b
        r = max(1.5, width / 2)
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length * r, dx / length * r
        polygon = [(ax + nx, ay + ny), (bx + nx, by + ny), (bx - nx, by - ny), (ax - nx, ay - ny)]
        gfx.filled_polygon(self.screen, polygon, color)
        gfx.aapolygon(self.screen, polygon, color)
        for cx, cy in ((ax, ay), (bx, by)):
            gfx.filled_circle(self.screen, int(cx), int(cy), int(r), color)
            gfx.aacircle(self.screen, int(cx), int(cy), int(r), color)

    def draw_fighter(self, fighter: Fighter) -> None:
        """The original look: a solid figure per player (P1 green, P2 pink), back limbs darker,
        the head outline white, yellow when hit, blue when blocking, the name under the feet."""
        if not fighter.points:
            return
        pg = self.pg
        s = {k: self.to_screen(*v) for k, v in fighter.points.items()}
        px = self.cam[2]
        f = fighter.state or {}
        color = PLAYER_COLORS[fighter.side]
        if fighter.heat_glow > 0.05:
            color = mix(color, (140, 225, 255), 0.35 * fighter.heat_glow)
        back = shade(color, 0.62)
        width = lambda metres: max(3, int(px * metres))
        # Contact shadow.
        sx, sy = self.to_screen(fighter.x, 0)
        pg.draw.ellipse(self.screen, (5, 7, 12), (sx - int(px * 0.28), sy - int(px * 0.035), int(px * 0.56), int(px * 0.07)))
        # Heat / Rage aura behind the body.
        for glow, aura in ((fighter.heat_glow, (40, 170, 255)), (fighter.rage_glow, (255, 30, 20))):
            if glow > 0.05:
                cx, cy = s["chest"]
                r = int(px * 0.9 * (0.9 + 0.1 * math.sin(self.frame_count * 0.2)))
                self.screen.blit(self.glow(max(4, r), aura, 0.6 * glow), (cx - r, cy - r), special_flags=pg.BLEND_ADD)
        near = "L" if fighter.facing > 0 else "R"   # the side toward the camera
        far = "R" if near == "L" else "L"
        # Back leg and arm first, so the torso and front limbs overlap them.
        self.limb(s["hips"], s["knee" + far], width(0.17), back)
        self.limb(s["knee" + far], s["ankle" + far], width(0.17), back)
        self.limb(s["shoulder"], s["elbow" + far], width(0.115), back)
        self.limb(s["elbow" + far], s["fist" + far], width(0.115), back)
        self.limb(s["hips"], s["chest"], width(0.31), color)  # torso
        hit = f.get("stun", 0) > 0 and not f.get("blockstun") and not f.get("move_id")
        blocking = bool(f.get("blockstun"))
        outline = (246, 224, 94) if hit else (99, 179, 237) if blocking else (237, 242, 247)
        hx, hy = s["head"]
        head = max(6, int(px * 0.19))
        self.gfx.filled_circle(self.screen, hx, hy, head, color)
        for ring in range(max(2, int(px * 0.03))):
            self.gfx.aacircle(self.screen, hx, hy, head - ring, outline)
        self.limb(s["hips"], s["knee" + near], width(0.17), color)
        self.limb(s["knee" + near], s["ankle" + near], width(0.17), color)
        self.limb(s["shoulder"], s["elbow" + near], width(0.115), color)
        self.limb(s["elbow" + near], s["fist" + near], width(0.115), color)
        # Strike trail.
        if len(fighter.trail) > 1:
            trail = (111, 224, 255) if f.get("heat") else (255, 60, 40) if f.get("rage") else (255, 255, 255)
            points = [self.to_screen(*p) for p in fighter.trail]
            for i in range(len(points) - 1):
                pg.draw.line(self.screen, trail, points[i], points[i + 1], max(1, int(width(0.08) * (1 - i / len(points)))))
        label = self.fonts["tiny"].render(title(fighter.slug), True, (226, 232, 240))
        self.screen.blit(label, label.get_rect(midtop=(sx, sy + int(px * 0.12))))

    def draw_effects(self) -> None:
        pg = self.pg
        px = self.cam[2]
        for x, y, life, maximum, size, color, kind in self.flashes:
            t = 1 - life / maximum
            sx, sy = self.to_screen(x, y)
            if kind == "star":
                r = max(4, int(px * size * 0.35 * (0.4 + ease(min(1, t * 3)) * 0.8)))
                self.screen.blit(self.glow(r, color, 1 - t), (sx - r, sy - r), special_flags=pg.BLEND_ADD)
                spikes, inner = 8, r * 0.18
                points = []
                for i in range(spikes * 2):
                    angle = i * math.pi / spikes + x * 7
                    radius = r * (1 - t * 0.5) if i % 2 == 0 else inner
                    points.append((sx + math.cos(angle) * radius, sy + math.sin(angle) * radius))
                if t < 0.6:
                    self.gfx.filled_polygon(self.screen, points, mix(color, (255, 255, 255), 0.6))
            elif kind == "ring":
                r = max(3, int(px * size * 0.5 * (0.2 + ease(t))))
                pg.draw.circle(self.screen, mix(color, (0, 0, 0), t), (sx, sy), r, max(1, int(6 * (1 - t))))
            elif kind == "floor":
                rx, ry = max(3, int(px * size * (0.2 + ease(t)))), max(2, int(px * size * 0.12 * (0.2 + ease(t))))
                pg.draw.ellipse(self.screen, mix(color, (0, 0, 0), t), (sx - rx, sy - ry, rx * 2, ry * 2),
                                max(1, int(5 * (1 - t))))
            elif kind == "shield":
                r = max(4, int(px * 0.32))
                surface = pg.Surface((r * 2 + 4, r * 2 + 4), pg.SRCALPHA)
                hexagon = [(r + 2 + math.cos(i * math.pi / 3) * r, r + 2 + math.sin(i * math.pi / 3) * r) for i in range(6)]
                pg.draw.polygon(surface, (*color, int(110 * (1 - t))), hexagon)
                pg.draw.polygon(surface, (*mix(color, (255, 255, 255), 0.5), int(255 * (1 - t))), hexagon, 3)
                self.screen.blit(surface, (sx - r - 2, sy - r - 2))
        for x, y, _, _, life, maximum, size, color, _ in self.particles:
            sx, sy = self.to_screen(x, y)
            r = max(1, int(px * size * 0.5))
            surface = self.glow(r + 2, color, min(1.0, life / (maximum * 0.5)))
            self.screen.blit(surface, (sx - r - 2, sy - r - 2), special_flags=pg.BLEND_ADD)

    def text(self, font: str, text: str, color, pos, anchor="topleft", outline=3, alpha=255) -> None:
        pg = self.pg
        render = self.fonts[font].render(text, True, color)
        shadow = self.fonts[font].render(text, True, (0, 0, 0))
        surface = pg.Surface((render.get_width() + outline * 2, render.get_height() + outline * 2), pg.SRCALPHA)
        for dx in range(-outline, outline + 1, max(1, outline)):
            for dy in range(-outline, outline + 1, max(1, outline)):
                surface.blit(shadow, (dx + outline, dy + outline))
        surface.blit(render, (outline, outline))
        if alpha < 255:
            surface.set_alpha(alpha)
        rect = surface.get_rect(**{anchor: pos})
        self.screen.blit(surface, rect)

    def draw_hud(self) -> None:
        pg = self.pg
        w, h = self.screen.get_size()
        if not self.current:
            self.text("caption", "Starting the fight...", (220, 220, 230), (w // 2, h // 2), "center")
            return
        bar_w, bar_h, top = int(w * 0.38), 26, 22
        for side, f in enumerate(self.current["fighters"]):
            fighter = self.fighters[side]
            x0 = 30 if side == 0 else w - 30 - bar_w
            health, recoverable = clamp(f["health"], 0, MAX_HEALTH), clamp(f["health"] + f["recoverable"], 0, MAX_HEALTH)
            self.lag[side] += (recoverable - self.lag[side]) * 0.04 if self.lag[side] > recoverable else recoverable - self.lag[side]
            slant = 14
            frame_poly = [(x0, top), (x0 + bar_w, top), (x0 + bar_w - slant, top + bar_h), (x0 - slant, top + bar_h)] \
                if side == 0 else [(x0, top), (x0 + bar_w, top), (x0 + bar_w + slant, top + bar_h), (x0 + slant, top + bar_h)]
            pg.draw.polygon(self.screen, (16, 16, 22), frame_poly)
            # Fill from the centre outward, like Tekken: P1 drains toward the left edge.
            def fill(amount, color):
                width = int(bar_w * amount / MAX_HEALTH)
                if width <= 0:
                    return
                if side == 0:
                    poly = [(x0 + bar_w - width, top), (x0 + bar_w, top), (x0 + bar_w - slant, top + bar_h),
                            (x0 + bar_w - width - slant, top + bar_h)]
                else:
                    poly = [(x0, top), (x0 + width, top), (x0 + width + slant, top + bar_h), (x0 + slant, top + bar_h)]
                pg.draw.polygon(self.screen, color, poly)
            fill(self.lag[side], (240, 240, 240))
            fill(recoverable, (140, 26, 22))
            low = health < MAX_HEALTH * 0.25
            fill(health, (232, 50, 43) if low else (255, 200, 60))
            fill(health * 0.999, (255, 120, 100) if low else (255, 215, 94))
            pg.draw.polygon(self.screen, (255, 60, 50) if f["rage"] else (201, 194, 168), frame_poly, 2)
            # Name, round wins, heat gauge, tags.
            name_pos = (x0, top + bar_h + 6) if side == 0 else (x0 + bar_w, top + bar_h + 6)
            self.text("name", title(fighter.slug).upper(), (244, 241, 234), name_pos, "topleft" if side == 0 else "topright")
            for i in range(ROUNDS_TO_WIN):
                cx = (x0 + bar_w - 20 - i * 24) if side == 0 else (x0 + 20 + i * 24)
                cy = top + bar_h + 22
                diamond = [(cx, cy - 8), (cx + 8, cy), (cx, cy + 8), (cx - 8, cy)]
                pg.draw.polygon(self.screen, (255, 215, 94) if i < self.wins[side] else (20, 20, 26), diamond)
                pg.draw.polygon(self.screen, (201, 194, 168), diamond, 2)
            gx, gy, gw = (x0, top + bar_h + 46, int(bar_w * 0.55)) if side == 0 else \
                (x0 + bar_w - int(bar_w * 0.55), top + bar_h + 46, int(bar_w * 0.55))
            pg.draw.rect(self.screen, (11, 22, 32), (gx, gy, gw, 8))
            if f["heat"]:
                amount = clamp(f["heat_frames"] / max(fighter.heat_max, 1), 0, 1)
                fill_x = gx if side == 1 else gx + gw - int(gw * amount)
                pg.draw.rect(self.screen, (79, 214, 255), (fill_x, gy, int(gw * amount), 8))
            elif f["heat_available"]:
                pg.draw.rect(self.screen, (40, 100, 140), (gx, gy, gw, 8))
            pg.draw.rect(self.screen, (60, 90, 110), (gx, gy, gw, 8), 1)
            tags = [t for t, on in (("HEAT", f["heat"]), ("RAGE", f["rage"])) if on]
            if tags:
                color = (79, 214, 255) if f["heat"] else (255, 50, 50)
                self.text("small", "  ".join(tags), color, (gx if side == 0 else gx + gw, gy + 12),
                          "topleft" if side == 0 else "topright", 2)
        # Timer.
        seconds = max(0, math.ceil((self.current["max_frames"] - self.current["frame"]) / FPS))
        box = pg.Rect(w // 2 - 50, top - 6, 100, 66)
        pg.draw.polygon(self.screen, (16, 16, 24), [(box.left + 12, box.top), (box.right - 12, box.top), (box.right, box.centery),
                                                     (box.right - 12, box.bottom), (box.left + 12, box.bottom), (box.left, box.centery)])
        pg.draw.polygon(self.screen, (201, 194, 168), [(box.left + 12, box.top), (box.right - 12, box.top), (box.right, box.centery),
                                                        (box.right - 12, box.bottom), (box.left + 12, box.bottom), (box.left, box.centery)], 2)
        self.text("timer", str(seconds), (250, 250, 250), box.center, "center", 2)
        self.text("tiny", f"ROUND {self.rounds_played + (0 if self.round_over else 1)}", (166, 163, 181),
                  (w // 2, box.bottom + 4), "midtop", 1)
        # Move captions.
        for side, caption in enumerate(self.captions):
            if caption and caption[3] > 0:
                alpha = int(255 * clamp(caption[3] / 20, 0, 1))
                anchor, x = ("bottomleft", 30) if side == 0 else ("bottomright", w - 30)
                y = h - 70
                self.text("caption", caption[0], (244, 241, 234), (x, y - 46), anchor, 2, alpha)
                self.text("cmd", caption[1], (255, 215, 94), (x, y - 22), anchor, 2, alpha)
                if caption[2]:
                    self.text("tiny", caption[2], (79, 214, 255), (x, y - 2), anchor, 1, alpha)
        # Combo counters.
        for side, (hits, damage, life) in enumerate(self.combo):
            if life > 0 and hits >= 2:
                alpha = int(255 * clamp(life / 20, 0, 1))
                anchor, x = ("topleft", 40) if side == 0 else ("topright", w - 40)
                self.text("combo", str(hits), (255, 215, 94), (x, int(h * 0.3)), anchor, 3, alpha)
                self.text("small", f"HITS   {int(damage)} DAMAGE", (230, 230, 240), (x, int(h * 0.3) + 60), anchor, 2, alpha)
        # Callouts.
        stack = [0, 0]
        for text, side, life, maximum, color in self.texts:
            t = 1 - life / maximum
            alpha = int(255 * clamp(min(t * 8, life / 15), 0, 1))
            x = int(w * 0.07) if side == 0 else int(w * 0.93)
            y = int(h * 0.22 - t * 20) + stack[side] * 44
            stack[side] += 1
            self.text("callout", text, color, (x, y), "topleft" if side == 0 else "topright", 3, alpha)
        # Big banner.
        if self.banner:
            text, life, maximum, color, small = self.banner
            t = 1 - life / maximum
            scale = 1 + max(0, 0.25 - t) * 5
            alpha = int(255 * clamp(life / 12, 0, 1))
            font = self.fonts["banner_small" if small else "banner"]
            render = font.render(text, True, color)
            outline = font.render(text, True, (0, 0, 0))
            surface = pg.Surface((render.get_width() + 10, render.get_height() + 10), pg.SRCALPHA)
            for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, 3), (-3, 3), (3, -3)):
                surface.blit(outline, (5 + dx, 5 + dy))
            surface.blit(render, (5, 5))
            highlight = font.render(text, True, (255, 255, 255))
            highlight.set_alpha(110)
            surface.blit(highlight, (5, 3), special_flags=pg.BLEND_RGBA_MAX)
            if scale != 1:
                surface = pg.transform.smoothscale(surface, (int(surface.get_width() * scale), int(surface.get_height() * scale)))
            surface.set_alpha(alpha)
            self.screen.blit(surface, surface.get_rect(center=(w // 2, int(h * 0.45))))
        # Screen flash.
        flash = getattr(self, "screen_flash", None)
        if flash:
            color, strength, life, maximum = flash
            overlay = pg.Surface((w, h))
            overlay.fill(color)
            overlay.set_alpha(int(255 * strength * life / maximum))
            self.screen.blit(overlay, (0, 0))
        # Rage Art vignette.
        if self.cinematic and self.cinematic["kind"] == "rage":
            overlay = pg.Surface((w, h), pg.SRCALPHA)
            for i in range(10):
                pg.draw.rect(overlay, (170, 0, 0, 14), (i * 12, i * 8, w - i * 24, h - i * 16), 14)
            self.screen.blit(overlay, (0, 0))
        # Footer and policy panel.
        match = self.match
        source = (f"{'Champion' if match.get('champion') else 'Checkpoint'} update {match['update']}"
                  if match.get("update") is not None else "Scripted fighters (no checkpoint yet)")
        mode = "greedy" if self.greedy else "sampled moves"
        footer = f"{source}  ·  {mode}  ·  {self.speed:g}x{'  ·  PAUSED' if self.paused else ''}" \
                 f"     Space pause  1/2/3 speed  N next  G greedy  P policy  Esc quit"
        self.text("tiny", footer, (166, 163, 181), (w // 2, h - 10), "midbottom", 1)
        if self.show_policy and self.current.get("choices"):
            table = self.moves.get(self.fighters[0].slug, {})
            panel = pg.Surface((300, 150), pg.SRCALPHA)
            panel.fill((8, 10, 18, 190))
            self.screen.blit(panel, (w - 330, 140))
            self.text("tiny", "P1 POLICY: TOP CHOICES", (166, 163, 181), (w - 318, 148), "topleft", 1)
            for i, (move, probability) in enumerate(self.current["choices"][0].get("top", [])[:5]):
                info = table.get(move)
                label = f"{info['command']}  {info['name']}" if info else move.replace("universal:", "")
                self.text("tiny", label[:30], (230, 230, 240), (w - 318, 172 + i * 22), "topleft", 1)
                self.text("tiny", f"{probability:.0%}", (255, 215, 94), (w - 40, 172 + i * 22), "topright", 1)

    def draw(self, dt: float) -> None:
        self.update_camera(dt)
        jitter = self.shake if self.shake > 0.3 else 0.0
        self.shake_offset = (self.random.uniform(-1, 1) * jitter, self.random.uniform(-1, 1) * jitter)
        self.draw_background()
        order = sorted(self.fighters, key=lambda f: 0 if (f.state and f.state.get("move_id")) else 1, reverse=True)
        for fighter in order:
            self.draw_fighter(fighter)
        self.draw_effects()
        self.draw_hud()

    def handle(self, event) -> bool:
        pg = self.pg
        if event.type == pg.QUIT:
            return False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                return False
            if event.key == pg.K_SPACE:
                self.paused = not self.paused
            elif event.key in (pg.K_1, pg.K_KP1):
                self.speed = 0.5
            elif event.key in (pg.K_2, pg.K_KP2):
                self.speed = 1.0
            elif event.key in (pg.K_3, pg.K_KP3):
                self.speed = 2.0
            elif event.key == pg.K_n:
                self.start_match()
            elif event.key == pg.K_g:
                self.greedy = not self.greedy
                self.start_match()
            elif event.key == pg.K_p:
                self.show_policy = not self.show_policy
        return True

    def run(self) -> None:
        pg = self.pg
        running = True
        while running:
            dt = min(0.1, self.clock.tick(FPS) / 1000)
            for event in pg.event.get():
                running = self.handle(event) and running
            self.step(dt)
            self.draw(dt)
            pg.display.flip()
        if self.feed:
            self.feed.close()
        pg.quit()

    def run_headless(self, frames: int) -> int:
        import time
        deadline = time.monotonic() + 60
        saved = 0
        while self.frame_count < frames and time.monotonic() < deadline:
            self.step(1 / 60)
            self.draw(1 / 60)
            shots = self.args.screenshots
            if shots and self.frame_count >= (saved + 1) * self.args.screenshot_every:
                saved += 1
                shots.mkdir(parents=True, exist_ok=True)
                self.pg.image.save(self.screen, str(shots / f"frame_{self.frame_count:05d}.png"))
        if self.feed:
            self.feed.close()
        self.pg.quit()
        print(f"visualizer headless: drew {self.frame_count} frames of {title(self.match['p1'])} vs "
              f"{title(self.match['p2'])}")
        return 0 if self.frame_count >= frames else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "train_all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--character")
    parser.add_argument("--opponent")
    parser.add_argument("--greedy", action="store_true", help="always the most likely move (stalls more)")
    parser.add_argument("--scripted-opponent", action="store_true", help="P2 is the scripted opponent")
    parser.add_argument("--seed", type=int, default=random.randrange(1 << 30))
    parser.add_argument("--headless-frames", type=int, help="render this many frames without a window, then exit")
    parser.add_argument("--screenshots", type=Path, help="with --headless-frames: save frames here")
    parser.add_argument("--screenshot-every", type=int, default=60)
    args = parser.parse_args()
    if not args.run.is_absolute():
        args.run = REPO / args.run
    if args.headless_frames:
        return App(args, headless=True).run_headless(args.headless_frames)
    App(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

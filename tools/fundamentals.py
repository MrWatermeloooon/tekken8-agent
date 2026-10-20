"""Does the bot grab, break grabs, counter-hit, and punish? Measured from real fights.

    python tools/fundamentals.py [--run runs/train_all | --checkpoint PATH] [--matchups 24] [--frames 7200]
                                 [--opponent-bot N]

The checkpoint plays itself (or, with --opponent-bot, a scripted bot) in random matchups through
t8_v2_full_move_feed with sampled play and its run's settings, and every frame is checked:

  grabs        how often it throws, how often a throw connects, and how often that throw
               is then broken
  breaks       when it is grabbed: how often it tries to break and how often it breaks,
               against what guessing a random break button would give for those throws
  counter-hits the share of its hits that are counter hits (hitting a move's startup)
  punishes     after blocking a move that is -10 or worse, how often it lands a hit before
               the attacker recovers; and after a whiff with 12+ frames of recovery left
               within 1.5 m, how often it hits during that recovery
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
# The memory/noise-capable feed while build-gpu is busy; either works for normal checkpoints.
FEEDS = [REPO / "build-gpu2" / "Release" / f"t8_v2_full_move_feed{EXE}", REPO / "build-gpu" / "Release" / f"t8_v2_full_move_feed{EXE}"]
THROW_LEVEL = 5
GUESS = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3, 4: 2 / 3}  # a random break button's chance per FullThrowBreak rule


def newest(directory: Path) -> Path | None:
    found = [p for p in directory.glob("update_*.t8ppo") if p.with_suffix(".t8fmstate").exists()]
    return max(found, key=lambda p: int(p.stem.split("_")[1])) if found else None


def trained_characters(checkpoint: Path) -> list[str]:
    data = checkpoint.with_suffix(".t8fmstate").read_bytes()
    offset, strings = 20, []
    for _ in range(3):
        length = int.from_bytes(data[offset:offset + 8], "little")
        strings.append(data[offset + 8:offset + 8 + length].decode("utf-8"))
        offset += 8 + length
    return [slug for slug in strings[2].split(",") if slug]


def play(feed: Path, checkpoint: Path, own: str, other: str, frames: int, seed: int, bot: int | None) -> Counter:
    command = [str(feed), "--character", own, "--opponent", other, "--checkpoint", str(checkpoint), "--sample",
               "--frames", str(frames), "--interval-ms", "0", "--seed", str(seed), "--allow-provisional"]
    command += ["--opponent-bot", str(bot)] if bot is not None else ["--opponent-checkpoint", str(checkpoint)]
    out = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout
    sides = [0] if bot is not None else [0, 1]  # whose play is measured
    c: Counter = Counter()
    previous = None
    move_key = [None, None]        # (move, start frame) of each fighter's current move
    blocked_seen = [False, False]  # its current move has been blocked
    whiff_seen = [False, False]
    punish_open: list[dict | None] = [None, None]  # an open opportunity for this defender
    break_window: list[dict | None] = [None, None]
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        f = json.loads(line)
        if previous is not None and f["round"] != previous["round"]:
            move_key, blocked_seen, whiff_seen = [None, None], [False, False], [False, False]
            punish_open, break_window = [None, None], [None, None]
        fighters = f["fighters"]
        distance = abs(fighters[0]["x"] - fighters[1]["x"])
        new_contacts = f["contacts"] if f["contact_age"] == 0 else []
        for p in (0, 1):
            me, them = fighters[p], fighters[1 - p]
            # Move starts.
            key = (me["move"], f["frame"] - me["action_frame"]) if me["move"] else None
            if key and key != move_key[p]:
                blocked_seen[p] = whiff_seen[p] = False
                if p in sides:
                    c["moves"] += 1
                    if me["hits"] and me["hits"][0][0] == THROW_LEVEL:
                        c["throws"] += 1
            move_key[p] = key
            # Punish opportunities for the other fighter (the defender).
            d = 1 - p
            if me["move"] and me["move_blocked"] and not blocked_seen[p]:
                blocked_seen[p] = True
                if me["block_advantage"] <= -10 and d in sides:
                    c["punish_chances"] += 1
                    punish_open[d] = {"kind": "punish", "until": f["frame"] + me["move_frames"] - me["action_frame"] + 1,
                                      "free_at": None, "tried": False}
            if me["move"] and me["hits"] and not me["move_hit"] and not me["move_blocked"] and not whiff_seen[p]:
                last_active = max(h[1] + h[2] - 1 for h in me["hits"])
                left = me["move_frames"] - me["action_frame"]
                if me["action_frame"] > last_active:
                    whiff_seen[p] = True
                    if left >= 12 and distance < 1.5 and d in sides:
                        c["whiff_chances"] += 1
                        punish_open[d] = {"kind": "whiff", "until": f["frame"] + left + 1, "free_at": None,
                                          "tried": False}
        for contact in new_contacts:
            a, kind = contact["attacker"], contact["contact"]
            if a in sides:
                c[f"contact:{kind}"] += 1
            if kind in ("hit", "counter_hit") and punish_open[a] and f["frame"] <= punish_open[a]["until"]:
                c[f"{punish_open[a]['kind']}_landed"] += 1
                punish_open[a] = None
            if kind == "throw_held":
                defender = 1 - a
                rule = fighters[a]["throw_rule"]
                if defender in sides and rule in GUESS:
                    break_window[defender] = {"rule": rule, "tried": False}
                    c["grabbed"] += 1
                    c["guess_expected"] += GUESS[rule]
            if kind == "throw_broken":
                defender = 1 - a
                if break_window[defender] is not None:
                    c["broke"] += 1
                    if not break_window[defender]["tried"]:
                        c["break_tries"] += 1  # pressed and broken on the same frame
                    break_window[defender] = None
        for p in (0, 1):
            chance = punish_open[p]
            if chance:
                me = fighters[p]
                if chance["free_at"] is None and me["actionable"]:
                    chance["free_at"] = f["frame"]
                    c[f"{chance['kind']}_free_frames"] += chance["until"] - f["frame"]  # frames left to land it
                if not chance["tried"] and me["move"] and me["action_frame"] <= 1:
                    chance["tried"] = True
                    c[f"{chance['kind']}_tried"] += 1
                    if chance["free_at"] is not None:
                        c[f"{chance['kind']}_reaction"] += f["frame"] - chance["free_at"]
                        c[f"{chance['kind']}_reacted"] += 1
            if punish_open[p] and f["frame"] > punish_open[p]["until"]:
                punish_open[p] = None
            window = break_window[p]
            if window is not None:
                if fighters[p]["break_attempted"] and not window["tried"]:
                    window["tried"] = True
                    c["break_tries"] += 1
                if fighters[p]["throw_break_frames"] == 0 and fighters[1 - p]["throw_rule"] < 0:
                    break_window[p] = None  # resolved without a break
        previous = f
    return c


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "train_all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--matchups", type=int, default=24)
    parser.add_argument("--frames", type=int, default=7200, help="frames per matchup (60 per second)")
    parser.add_argument("--opponent-bot", type=int, help="a scripted bot id instead of self-play")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    checkpoint = args.checkpoint or newest(run / "champion") or newest(run / "checkpoints")
    if checkpoint is None:
        raise SystemExit(f"no checkpoint in {run}")
    feed = next((f for f in FEEDS if f.exists()), None)
    if feed is None:
        raise SystemExit("build t8_v2_full_move_feed first")
    roster = trained_characters(checkpoint)
    rng = random.Random(args.seed)
    jobs = [(rng.choice(roster), rng.choice(roster), rng.randrange(1 << 30)) for _ in range(args.matchups)]
    with ThreadPoolExecutor(max(1, min(8, (os.cpu_count() or 4) - 2))) as pool:
        results = list(pool.map(lambda j: play(feed, checkpoint, j[0], j[1], args.frames, j[2], args.opponent_bot), jobs))
    c = sum(results, Counter())
    rate = lambda a, b: f"{c[a] / c[b]:.0%} ({c[a]:,} of {c[b]:,})" if c[b] else "n/a (none)"
    hits = c["contact:hit"] + c["contact:counter_hit"]
    print(f"{checkpoint} vs {'bot ' + str(args.opponent_bot) if args.opponent_bot is not None else 'itself'}, "
          f"{args.matchups} matchups x {args.frames} frames, sampled play\n")
    print("Grabs")
    print(f"  throws: {rate('throws', 'moves')} of its moves")
    print(f"  connected: {c['contact:throw_held']:,}   then broken by the opponent: {c['contact:throw_broken']:,}")
    print("Breaking grabs (when it is grabbed)")
    print(f"  tried to break: {rate('break_tries', 'grabbed')}")
    print(f"  broke:          {rate('broke', 'grabbed')}   (random button guessing would break "
          f"{c['guess_expected'] / max(c['grabbed'], 1):.0%})")
    print("Counter-attacks")
    print(f"  counter hits: {c['contact:counter_hit'] / max(hits, 1):.0%} of its {hits:,} hits")
    print(f"  punishes after blocking a -10 or worse move: {rate('punish_landed', 'punish_chances')}")
    print(f"    attacked at all before the attacker recovered: {rate('punish_tried', 'punish_chances')}")
    if c["punish_reacted"]:
        print(f"    frames from able-to-act to attacking: {c['punish_reaction'] / c['punish_reacted']:.1f} on average; "
              f"frames it had left: {c['punish_free_frames'] / max(c['punish_chances'], 1):.1f}")
    print(f"  whiff punishes (12+ frames of recovery, within 1.5 m): {rate('whiff_landed', 'whiff_chances')}")
    print(f"    attacked at all during that recovery: {rate('whiff_tried', 'whiff_chances')}")
    other = {k.split(':')[1]: v for k, v in c.items() if k.startswith("contact:")}
    print(f"\nall its contacts: {dict(sorted(other.items(), key=lambda kv: -kv[1]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

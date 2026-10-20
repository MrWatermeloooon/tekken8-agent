"""Which moves a full-move checkpoint actually uses, per character.

    python tools/move_usage.py [--checkpoint PATH | --run runs/train_all] [--characters all|jun,king]
                               [--frames 15000] [--opponent self|scripted] [--top 5]

For each character, one CPU fight (t8_v2_full_move_feed, greedy policy) runs for
--frames frames against the same checkpoint (self) or the scripted opponent, and
every move the learner starts is counted. Without --checkpoint, the newest
champion of --run is used (else its newest checkpoint).

Per character: moves started, distinct moves, "effective moves" (exp of the
usage entropy: 10 means as varied as ten equally used moves), the top moves'
share, and the top moves. A policy that spams a few buttons shows a small
effective count and a top-3 share near 100%. Writes <run>/move_usage.md and .csv.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""


def newest(directory: Path) -> Path | None:
    found = [p for p in directory.glob("update_*.t8ppo") if p.with_suffix(".t8fmstate").exists()]
    return max(found, key=lambda p: int(p.stem.split("_")[1])) if found else None


def trained_characters(checkpoint: Path) -> list[str]:
    # .t8fmstate: "T8FMV1" magic (8 bytes), u32 version, u64 update, then
    # u64-length-prefixed strings: catalog, roster, learners (comma-separated).
    data = checkpoint.with_suffix(".t8fmstate").read_bytes()
    offset = 8 + 4 + 8
    strings = []
    for _ in range(3):
        length = int.from_bytes(data[offset:offset + 8], "little")
        offset += 8
        strings.append(data[offset:offset + length].decode("utf-8"))
        offset += length
    return [slug for slug in strings[2].split(",") if slug]


SAMPLE = False  # --sample: moves drawn from the policy as in training (default greedy)


def usage_of(feed: str, checkpoint: Path, character: str, opponent: str, frames: int, seed: int,
             self_play: bool) -> Counter:
    command = [feed, "--character", character, "--opponent", opponent, "--checkpoint", str(checkpoint),
               "--frames", str(frames), "--interval-ms", "0", "--seed", str(seed), "--allow-provisional"]
    if SAMPLE:
        command.append("--sample")
    if self_play:
        command += ["--opponent-checkpoint", str(checkpoint)]
    started: Counter = Counter()
    previous_move, previous_frame = "", -1
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace")
    assert process.stdout is not None
    for line in process.stdout:
        if not line.startswith("{"):
            continue
        fighter = json.loads(line)["fighters"][0]
        move, frame = fighter["move"], fighter["action_frame"]
        if move and (move != previous_move or frame < previous_frame):
            started[move] += 1
        previous_move, previous_frame = move, frame
    if process.wait() != 0:
        raise RuntimeError(f"feed failed for {character}: {process.stderr.read() if process.stderr else ''}")
    return started


def notation() -> dict[str, str]:
    names = {}
    with (REPO / "data" / "generated" / "full_move_catalog.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names[row["stable_id"]] = row["command"] or row["name"]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "train_all")
    parser.add_argument("--characters", default="all")
    parser.add_argument("--frames", type=int, default=15000)
    parser.add_argument("--opponent", choices=("self", "scripted"), default="self")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample", action="store_true", help="sample moves as in training (default: greedy)")
    args = parser.parse_args()
    global SAMPLE
    SAMPLE = args.sample
    feed = next((str(REPO / d / f"t8_v2_full_move_feed{EXE}") for d in ("build-gpu/Release", "build-gpu")
                 if (REPO / d / f"t8_v2_full_move_feed{EXE}").exists()), None)
    if feed is None:
        raise SystemExit("t8_v2_full_move_feed not built")
    checkpoint = args.checkpoint or newest(args.run / "champion") or newest(args.run / "checkpoints")
    if checkpoint is None:
        raise SystemExit(f"no checkpoint with a .t8fmstate in {args.run}")
    learned = trained_characters(checkpoint)
    characters = learned if args.characters == "all" else args.characters.split(",")
    if not characters:
        raise SystemExit("could not read the checkpoint's characters; pass --characters")
    print(f"checkpoint: {checkpoint}  ({len(characters)} characters, {args.frames} frames each, vs {args.opponent})",
          flush=True)
    names = notation()
    label = lambda move: f"{names[move]} ({move.split(':')[1]})" if move in names else move
    rows = []
    for index, character in enumerate(characters):
        opponent = characters[(index * 7 + 3) % len(characters)]
        counts = usage_of(feed, checkpoint, character, opponent, args.frames, args.seed + index, args.opponent == "self")
        total = sum(counts.values())
        if total == 0:
            continue
        entropy = -sum(c / total * math.log(c / total) for c in counts.values())
        ranked = counts.most_common()
        top3 = sum(c for _, c in ranked[:3]) / total
        rows.append({"character": character, "opponent": opponent, "moves_started": total, "distinct": len(counts),
                     "effective": math.exp(entropy), "top3_share": top3,
                     "top": "; ".join(f"{label(move)} {c / total:.0%}" for move, c in ranked[:args.top])})
        r = rows[-1]
        print(f"{character:>12}: {r['distinct']:3d} distinct, effective {r['effective']:5.1f}, top-3 {top3:4.0%}  "
              f"| {r['top']}", flush=True)
    output = (args.run if args.checkpoint is None else checkpoint.parent)
    with (output / "move_usage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    effective = sorted(r["effective"] for r in rows)
    summary = (f"median effective moves {effective[len(effective) // 2]:.1f}, lowest {effective[0]:.1f} "
               f"({min(rows, key=lambda r: r['effective'])['character']}), "
               f"median top-3 share {sorted(r['top3_share'] for r in rows)[len(rows) // 2]:.0%}")
    lines = [f"# Move usage: {checkpoint.name}", "", f"{args.frames} frames per character vs {args.opponent}; {summary}.",
             "", "| character | moves started | distinct | effective | top-3 share | top moves |",
             "|---|---:|---:|---:|---:|---|"]
    lines += [f"| {r['character']} | {r['moves_started']} | {r['distinct']} | {r['effective']:.1f} | "
              f"{r['top3_share']:.0%} | {r['top']} |" for r in rows]
    (output / "move_usage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

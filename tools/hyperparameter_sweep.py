"""Equal-budget hyperparameter comparison for the full-move trainer.

    python tools/hyperparameter_sweep.py [--steps 400000000] [--only baseline,hidden512] [--rounds 2048]

Every variant trains the one all-character policy from the same seed for the
same number of environment steps (not updates: a variant with twice the
environments gets half the updates). The final checkpoints then play a mixed
all-character round robin (t8_v2_full_move_arena --mixed), which is the score
that matters; the scripted held-out win rate saturates too early to rank them.

Results: runs/sweep/summary.csv and summary.md (arena rating, held-out win rate,
worst learner, entropy, throughput, wall time). A variant that already finished
is not trained again, so an interrupted sweep continues where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""

# The T8Train defaults (training.ini): 32768 envs x 64 steps = 2.1M steps per update.
BASELINE = {
    "envs": 32768, "horizon": 64, "minibatch": 131072, "epochs": 4, "hidden": 256,
    "learning-rate": 3e-4, "final-learning-rate": 3e-5, "entropy-target": 1.0, "target-kl": 0.02,
    "matmul": "fp32",
}
VARIANTS: dict[str, dict[str, float | str]] = {
    "baseline": {},
    "hidden512": {"hidden": 512},
    "lr1.5e-4": {"learning-rate": 1.5e-4, "final-learning-rate": 1.5e-5},
    "epochs2": {"epochs": 2},
    "minibatch32k": {"minibatch": 32768},
    "entropy1.5": {"entropy-target": 1.5},
    "envs65536": {"envs": 65536, "minibatch": 262144},
    # Tensor-core matrix multiplies (FP32 storage and optimizer either way).
    "mb32k_hidden512": {"minibatch": 32768, "hidden": 512},
    "tf32": {"matmul": "tf32"},
    "bf16": {"matmul": "bf16"},
}


def binary(name: str) -> str:
    for directory in ("build-gpu/Release", "build-gpu"):
        path = REPO / directory / f"{name}{EXE}"
        if path.exists():
            return str(path)
    raise SystemExit(f"{name}{EXE} not found: build build-gpu first")


def final_checkpoint(run: Path) -> Path | None:
    found = sorted(run.glob("checkpoints/update_*.t8ppo"), key=lambda p: int(p.stem.split("_")[1]))
    return found[-1] if found else None


def train(name: str, settings: dict, updates: int, seed: int, run: Path) -> float:
    metrics = run / "metrics.jsonl"
    if metrics.exists() and metrics.read_text(encoding="utf-8").strip():
        last = json.loads(metrics.read_text(encoding="utf-8").strip().splitlines()[-1])
        if last.get("update") == updates:
            print(f"{name}: already trained", flush=True)
            return float(last.get("elapsed_seconds", 0.0))
    command = [binary("t8_v2_full_move_train"), "--learner-character", "all", "--opponents", "roster",
               "--allow-provisional", "--seed", str(seed), "--updates", str(updates), "--run-dir", str(run),
               "--checkpoint-interval", str(max(1, updates // 4)), "--eval-interval", str(max(1, updates // 4)),
               "--champion-interval-steps", "0", "--self-play-share", "0.3",
               "--self-play-start", str(max(1, updates // 15)), "--scenario-share", "0.25"]
    for key, value in settings.items():
        command += [f"--{key}", f"{value:g}" if isinstance(value, float) else str(value)]
    if (run / "checkpoints").exists():
        command.append("--resume")
    print(f"{name}: training {updates} updates  ({' '.join(command[1:])})", flush=True)
    started = time.perf_counter()
    run.mkdir(parents=True, exist_ok=True)
    with (run / "sweep.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed (exit {result.returncode}); see {run / 'sweep.log'}")
    return time.perf_counter() - started


def summarize_run(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    tail = rows[-max(1, len(rows) // 10):]  # the last tenth, so one noisy update does not decide
    evaluated = [row["evaluation"] for row in rows if "evaluation" in row]
    learners = run / "evaluation_learners.csv"
    worst = ""
    if learners.exists():
        table = list(csv.DictReader(learners.open(encoding="utf-8")))
        last_update = table[-1]["update"] if table else None
        final = [row for row in table if row["update"] == last_update and row.get("policy", "deterministic") == "deterministic"]
        if final:
            low = min(final, key=lambda row: float(row["win_rate"]))
            worst = f"{low['learner_character']} {float(low['win_rate']):.2f}"
    return {
        "steps": rows[-1]["environment_steps"],
        "wall_minutes": rows[-1]["elapsed_seconds"] / 60.0,
        "steps_per_second": rows[-1]["environment_steps"] / max(rows[-1]["elapsed_seconds"], 1e-9),
        "entropy": sum(row["rollout_decision_entropy"] for row in tail) / len(tail),
        "kl": sum(row["approximate_kl"] for row in tail) / len(tail),
        "kl_stops": sum(bool(row.get("kl_early_stop")) for row in rows) / len(rows),
        "heldout_win_rate": evaluated[-1]["win_rate"] if evaluated else float("nan"),
        "worst_learner": worst,
    }


def arena(names: list[str], checkpoints: list[Path], rounds: int, seed: int, output: Path) -> dict[str, float]:
    command = [binary("t8_v2_full_move_arena"), "--mixed", "--allow-provisional", "--rounds-per-pair", str(rounds),
               "--seed", str(seed), "--output", str(output), *map(str, checkpoints)]
    print("arena: " + " vs ".join(names), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    # Bradley-Terry ratings from the pairwise results (draws count half), anchored at baseline = 1500.
    score = [[0.0] * len(names) for _ in names]
    games = [[0.0] * len(names) for _ in names]
    for row in csv.DictReader(output.open(encoding="utf-8")):
        a, b = int(row["a"]), int(row["b"])
        wins, losses, draws = int(row["wins_a"]), int(row["wins_b"]), int(row["draws"])
        total = wins + losses + draws
        score[a][b] += wins + 0.5 * draws
        score[b][a] += losses + 0.5 * draws
        games[a][b] += total
        games[b][a] += total
    strength = [1.0] * len(names)
    for _ in range(2000):
        for i in range(len(names)):
            # One virtual draw against each opponent keeps a clean sweep finite.
            won = sum(score[i][j] + 0.5 for j in range(len(names)) if j != i)
            expected = sum((games[i][j] + 1.0) / (strength[i] + strength[j]) for j in range(len(names)) if j != i)
            strength[i] = won / expected
        anchor = strength[names.index("baseline")] if "baseline" in names else strength[0]
        strength = [value / anchor for value in strength]
    return {name: 1500.0 + 400.0 * math.log10(value) for name, value in zip(names, strength)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=float, default=4e8, help="environment steps per variant")
    parser.add_argument("--only", default="", help="comma list of variants (always adds baseline)")
    parser.add_argument("--rounds", type=int, default=2048, help="arena rounds per pairing")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--variant", action="append", default=[], metavar="NAME:key=value,...",
                        help="an extra variant, e.g. mb16k:minibatch=16384,matmul=tf32 (repeatable)")
    parser.add_argument("--base", default="", metavar="key=value,...",
                        help="override the baseline itself (every variant builds on it)")
    parser.add_argument("--directory", type=Path, default=REPO / "runs" / "sweep")
    args = parser.parse_args()

    def settings_of(text: str) -> dict:
        parsed = {}
        for pair in filter(None, text.split(",")):
            key, value = pair.split("=", 1)
            try:
                parsed[key] = int(value)
            except ValueError:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        return parsed

    BASELINE.update(settings_of(args.base))
    for spec in args.variant:
        name, _, text = spec.partition(":")
        VARIANTS[name] = settings_of(text)
        args.only = ",".join(filter(None, [args.only, name]))
    chosen = [name for name in VARIANTS if not args.only or name in args.only.split(",") or name == "baseline"]
    unknown = set(filter(None, args.only.split(","))) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(sorted(unknown))}; known: {', '.join(VARIANTS)}")
    args.directory.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    checkpoints: list[Path] = []
    for name in chosen:
        settings = {**BASELINE, **VARIANTS[name]}
        per_update = settings["envs"] * settings["horizon"]
        updates = max(4, round(args.steps / per_update))
        run = args.directory / name
        train(name, settings, updates, args.seed, run)
        summaries[name] = {"variant": name, **{k: v for k, v in VARIANTS[name].items()}, **summarize_run(run)}
        checkpoint = final_checkpoint(run)
        if checkpoint is None:
            raise SystemExit(f"{name} wrote no checkpoint")
        checkpoints.append(checkpoint)
    ratings = arena(chosen, checkpoints, args.rounds, args.seed, args.directory / "arena.csv") if len(chosen) > 1 else {}
    ordered = sorted(chosen, key=lambda name: -ratings.get(name, 1500.0))
    columns = ["variant", "arena_rating", "heldout_win_rate", "worst_learner", "entropy", "kl", "kl_stops",
               "steps_per_second", "wall_minutes", "steps"]
    with (args.directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for name in ordered:
            writer.writerow({**summaries[name], "arena_rating": round(ratings.get(name, 1500.0), 1)})
    lines = ["| variant | arena rating | held-out win | worst learner | entropy | KL | KL stops | steps/s | minutes |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name in ordered:
        s = summaries[name]
        lines.append(f"| {name} | {ratings.get(name, 1500.0):.0f} | {s['heldout_win_rate']:.3f} | {s['worst_learner']} | "
                     f"{s['entropy']:.2f} | {s['kl']:.4f} | {s['kl_stops']:.0%} | {s['steps_per_second']:,.0f} | "
                     f"{s['wall_minutes']:.1f} |")
    (args.directory / "summary.md").write_text(
        f"# Hyperparameter sweep ({args.steps:,.0f} steps per variant, seed {args.seed})\n\n"
        "Arena rating: mixed all-character round robin of the final checkpoints, baseline = 1500.\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

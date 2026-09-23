"""Aggregates held-out evaluation results across seeds of one training recipe.

Reads each run's evaluations.csv (written by t8_v2_train, or backfilled with
tools/export_evaluations.py) and reports per-seed and across-seed statistics:
final performance, peak, worst drawdown after the peak, stability, side
balance, and regression-guard actions. A learning-quality claim needs at
least five seeds that all reached the target update; the report states
whether that bar is met.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MINIMUM_SEEDS = 5
MINIMUM_LONG_UPDATES = 10_000  # below this a run is a smoke test, not "long training"
FINAL_WINDOW = 10  # evaluations averaged for the "final" score
STABLE_THRESHOLD = 0.80


def load(run_directory: Path) -> list[dict]:
    path = run_directory / "evaluations.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["update"] = int(row["update"])
        for key in ("win_rate", "p1_win_rate", "p2_win_rate", "worst_style_win_rate", "stochastic_win_rate"):
            row[key] = float(row[key])
    return rows


def per_seed(run_directory: Path, rows: list[dict]) -> dict:
    rates = [row["win_rate"] for row in rows]
    final = rates[-FINAL_WINDOW:]
    peak_index = max(range(len(rates)), key=rates.__getitem__)
    running_peak, drawdown = rates[0], 0.0
    for rate in rates:
        running_peak = max(running_peak, rate)
        drawdown = max(drawdown, running_peak - rate)
    first_85 = next((row["update"] for row in rows if row["win_rate"] >= 0.85), None)
    return {
        "run_directory": str(run_directory),
        "evaluations": len(rows),
        "last_update": rows[-1]["update"],
        "final_win_rate": statistics.fmean(final),
        "final_stochastic_win_rate": statistics.fmean(row["stochastic_win_rate"] for row in rows[-FINAL_WINDOW:]),
        "final_side_gap": statistics.fmean(abs(row["p1_win_rate"] - row["p2_win_rate"])
                                           for row in rows[-FINAL_WINDOW:]),
        "final_worst_style_win_rate": statistics.fmean(row["worst_style_win_rate"] for row in rows[-FINAL_WINDOW:]),
        "peak_win_rate": rates[peak_index],
        "peak_update": rows[peak_index]["update"],
        "minimum_win_rate": min(rates),
        "max_drawdown": drawdown,
        "fraction_at_or_above_0_80": sum(rate >= STABLE_THRESHOLD for rate in rates) / len(rates),
        "first_update_at_or_above_0_85": first_85,
        "guard_rollbacks": sum(row.get("guard_action") == "rollback" for row in rows),
        "guard_pauses": sum(row.get("guard_action") == "pause" for row in rows),
    }


def across(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    result = {"n": len(values), "median": statistics.median(values), "mean": mean,
              "min": min(values), "max": max(values)}
    if len(values) > 1:
        sd = statistics.stdev(values)
        # Two-sided 95% Student-t critical values for small n.
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
             10: 2.262}.get(len(values), 1.96)
        result.update(sd=sd, ci95_low=mean - t * sd / math.sqrt(len(values)),
                      ci95_high=mean + t * sd / math.sqrt(len(values)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--target-update", type=int, default=None,
                        help="Update every seed must reach (default: the largest reached).")
    parser.add_argument("--minimum-long-updates", type=int, default=MINIMUM_LONG_UPDATES,
                        help="Shortest target that counts as long training for a learning-quality claim.")
    parser.add_argument("--output", type=Path, required=True, help="Output path stem (.json and .md are added).")
    args = parser.parse_args()

    seeds, missing = [], []
    for directory in args.run_directories:
        rows = load(directory)
        (seeds.append(per_seed(directory, rows)) if rows else missing.append(str(directory)))
    if not seeds:
        raise SystemExit("no run has evaluations.csv yet")
    target = args.target_update or max(seed["last_update"] for seed in seeds)
    complete = [seed for seed in seeds if seed["last_update"] >= target]
    metrics = ("final_win_rate", "final_stochastic_win_rate", "final_side_gap", "final_worst_style_win_rate",
               "peak_win_rate", "minimum_win_rate", "max_drawdown", "fraction_at_or_above_0_80")
    summary = {metric: across([seed[metric] for seed in complete]) for metric in metrics} if complete else {}
    long_enough = target >= args.minimum_long_updates
    claim_ready = (long_enough and len(complete) >= MINIMUM_SEEDS and not missing
                   and len(complete) == len(seeds))
    report = {"target_update": target, "minimum_long_updates": args.minimum_long_updates, "seeds": seeds, "missing": missing,
              "complete_seeds": len(complete), "summary_over_complete_seeds": summary,
              "learning_quality_claim_ready": claim_ready}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# Seed matrix report (target update {target})", "",
             f"Complete seeds: {len(complete)}/{len(seeds) + len(missing)}. "
             f"Learning-quality claim ready (>= {MINIMUM_SEEDS} complete seeds, none missing, target >= "
             f"{args.minimum_long_updates} updates): **{'yes' if claim_ready else 'no'}**"
             + ("" if long_enough else f" (target {target} is a smoke-length run)") + ".", "",
             "| Run | Last update | Final (last 10) | Peak (update) | Min | Max drawdown | >=80% | Side gap | "
             "Worst style | Rollbacks / pauses |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        lines.append(
            f"| {Path(seed['run_directory']).name} | {seed['last_update']} | {seed['final_win_rate']:.3f} | "
            f"{seed['peak_win_rate']:.3f} ({seed['peak_update']}) | {seed['minimum_win_rate']:.3f} | "
            f"{seed['max_drawdown']:.3f} | {seed['fraction_at_or_above_0_80']:.0%} | {seed['final_side_gap']:.3f} | "
            f"{seed['final_worst_style_win_rate']:.3f} | {seed['guard_rollbacks']} / {seed['guard_pauses']} |")
    if summary:
        lines += ["", "Across complete seeds:", "", "| Metric | Median | Mean (95% CI) | Range |", "|---|---:|---:|---:|"]
        for metric, stats in summary.items():
            interval = (f"{stats['mean']:.3f} ({stats['ci95_low']:.3f} to {stats['ci95_high']:.3f})"
                        if "ci95_low" in stats else f"{stats['mean']:.3f}")
            lines.append(f"| {metric} | {stats['median']:.3f} | {interval} | {stats['min']:.3f} to {stats['max']:.3f} |")
    if missing:
        lines += ["", "Runs without evaluations yet: " + ", ".join(missing)]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

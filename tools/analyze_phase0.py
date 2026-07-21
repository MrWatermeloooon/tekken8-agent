from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else math.nan,
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
    }


def normalized_auc(rows: list[dict[str, Any]]) -> float:
    points = [
        (float(row["environment_steps"]), float(row["evaluation"]["total"]["win_rate"]))
        for row in rows
        if "evaluation" in row
    ]
    if not points:
        raise ValueError("run has no evaluation rows")
    points.sort()
    if len(points) == 1:
        return points[0][1]
    area = 0.0
    previous_x, previous_y = 0.0, points[0][1]
    for current_x, current_y in points:
        area += (current_x - previous_x) * (previous_y + current_y) * 0.5
        previous_x, previous_y = current_x, current_y
    return area / points[-1][0]


def load_run(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or "evaluation" not in rows[-1]:
        raise ValueError(f"final metrics row has no evaluation: {path}")
    updates = [int(row["update"]) for row in rows]
    if updates != list(range(1, updates[-1] + 1)):
        raise ValueError(f"metrics updates are incomplete or unordered: {path}")
    final = rows[-1]
    evaluation = final["evaluation"]
    return {
        "path": str(path),
        "updates": updates[-1],
        "environment_steps": int(final["environment_steps"]),
        "elapsed_seconds": float(final["elapsed_seconds"]),
        "final_win_rate": float(evaluation["total"]["win_rate"]),
        "auc": normalized_auc(rows),
        "as_p1_win_rate": float(evaluation["as_p1"]["win_rate"]),
        "as_p2_win_rate": float(evaluation["as_p2"]["win_rate"]),
        "draw_rate": float(evaluation["total"]["draws"]) / float(evaluation["total"]["episodes"]),
        "timeout_rate": float(evaluation["total"]["timeouts"]) / float(evaluation["total"]["episodes"]),
        "mean_damage_dealt": float(evaluation["total"]["mean_damage_dealt"]),
        "mean_damage_taken": float(evaluation["total"]["mean_damage_taken"]),
        "style_win_rates": [float(style["win_rate"]) for style in evaluation["total"]["styles"]],
        "final_kl": float(final["approximate_kl"]),
        "final_entropy": float(final["entropy"]),
        "final_gradient_norm": float(final["gradient_norm"]),
        "final_value_loss": float(final["value_loss"]),
        "observation_mode": final.get("observation_mode", "legacy-unspecified"),
        "benchmark": final.get("benchmark", "legacy-unspecified"),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "final_win_rate", "auc", "elapsed_seconds", "as_p1_win_rate", "as_p2_win_rate",
        "draw_rate", "timeout_rate", "mean_damage_dealt", "mean_damage_taken", "final_kl",
        "final_entropy", "final_gradient_norm", "final_value_loss",
    ]
    result = {field: summary([float(run[field]) for run in runs]) for field in fields}
    result["style_win_rates"] = [
        summary([run["style_win_rates"][style] for run in runs]) for style in range(8)
    ]
    result["runs"] = runs
    return result


def render_markdown(label: str, report: dict[str, Any]) -> str:
    def cell(metric: dict[str, Any]) -> str:
        return f"{metric['median']:.3f} [{metric['q1']:.3f}, {metric['q3']:.3f}]"

    lines = [
        f"# Phase 0 report: {label}",
        "",
        "All policy selection numbers use the held-out V2 suite, balanced across P1/P2 and eight styles.",
        "Values are median [Q1, Q3] across seeds.",
        "",
        "| Metric | Shaped | Sparse |",
        "|---|---:|---:|",
    ]
    labels = {
        "final_win_rate": "Final win rate",
        "auc": "Normalized win-rate AUC",
        "as_p1_win_rate": "P1 win rate",
        "as_p2_win_rate": "P2 win rate",
        "draw_rate": "Draw rate",
        "timeout_rate": "Timeout rate",
        "mean_damage_dealt": "Mean damage dealt",
        "mean_damage_taken": "Mean damage taken",
        "elapsed_seconds": "Wall time (seconds)",
        "final_kl": "Final approximate KL",
        "final_entropy": "Final entropy",
        "final_gradient_norm": "Final gradient norm",
        "final_value_loss": "Final value loss",
    }
    for key, label_text in labels.items():
        lines.append(f"| {label_text} | {cell(report['shaped'][key])} | {cell(report['sparse'][key])} |")
    lines += ["", "## Held-out style win rates", "", "| Style | Shaped | Sparse |", "|---:|---:|---:|"]
    for style in range(8):
        lines.append(
            f"| {style} | {cell(report['shaped']['style_win_rates'][style])} | "
            f"{cell(report['sparse']['style_win_rates'][style])} |"
        )
    lines += ["", "## Gate decision", "", report["gate_decision"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate held-out, side-balanced V2 Phase 0 runs.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = {"shaped": [], "sparse": []}
    for reward in grouped:
        paths = sorted(args.runs_root.glob(f"{args.label}_{reward}_seed*/metrics.jsonl"))
        if len(paths) < 3:
            raise SystemExit(f"need at least three {reward} runs for {args.label}; found {len(paths)}")
        grouped[reward] = [load_run(path) for path in paths]
    modes = {run["observation_mode"] for runs in grouped.values() for run in runs}
    benchmarks = {run["benchmark"] for runs in grouped.values() for run in runs}
    if len(modes) != 1 or len(benchmarks) != 1:
        raise SystemExit("refusing to aggregate mixed observation modes or benchmark protocols")

    report: dict[str, Any] = {
        "label": args.label,
        "observation_mode": next(iter(modes)),
        "benchmark": next(iter(benchmarks)),
        "shaped": aggregate(grouped["shaped"]),
        "sparse": aggregate(grouped["sparse"]),
    }
    shaped = report["shaped"]["final_win_rate"]
    sparse = report["sparse"]["final_win_rate"]
    if shaped["q1"] > sparse["q3"] and shaped["median"] - sparse["median"] >= 0.05:
        decision = "Shaping shows a reproducible advantage; investigate return redistribution next."
    else:
        decision = "No promotion-quality shaped-reward advantage was established; keep return redistribution gated."
    report["gate_decision"] = decision

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(args.label, report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

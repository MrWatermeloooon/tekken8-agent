"""Backfills evaluations.csv and evaluation_styles.csv from a run's metrics.jsonl.

Runs trained since the evaluation exports landed get these files written
directly by t8_v2_train at every evaluation. This tool produces the same
columns for older runs. Per-character results were never recorded in those
ledgers, so evaluation_characters.csv cannot be backfilled; use
`t8_v2_train --probe-checkpoint` on individual checkpoints instead.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUMMARY_HEADER = [
    "update", "environment_steps", "elapsed_seconds", "episodes", "training_opponent",
    "latest_checkpoint_update", "best_older_checkpoint_update",
    "win_rate", "p1_win_rate", "p2_win_rate", "draw_rate", "worst_style_win_rate",
    "mean_damage_dealt", "mean_damage_taken",
    "stochastic_win_rate", "stochastic_p1_win_rate", "stochastic_p2_win_rate", "stochastic_draw_rate",
    "stochastic_worst_style_win_rate", "stochastic_mean_damage_dealt", "stochastic_mean_damage_taken",
    "guard_action",
]
STYLE_HEADER = ["update", "policy", "side", "style", "episodes", "wins", "losses", "draws", "win_rate"]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summary(evaluation: dict) -> list:
    total = evaluation["total"]
    styles = [style["win_rate"] for style in total.get("styles", []) if style.get("episodes")]
    return [
        _rate(total["wins"], total["episodes"]),
        evaluation["as_p1"]["win_rate"],
        evaluation["as_p2"]["win_rate"],
        _rate(total["draws"], total["episodes"]),
        min(styles) if styles else 0.0,
        total["mean_damage_dealt"],
        total["mean_damage_taken"],
    ]


def export(run_directory: Path, output_directory: Path) -> tuple[int, int]:
    summary_rows, style_rows = [], []
    with (run_directory / "metrics.jsonl").open(encoding="utf-8") as ledger:
        for line in ledger:
            record = json.loads(line)
            deterministic = record.get("evaluation")
            if not deterministic:
                continue
            stochastic = record.get("evaluation_stochastic", deterministic)
            self_play = str(record.get("training_opponent", "")).startswith("self_play")
            guard = record.get("regression_guard", {}).get("action", "")
            summary_rows.append([
                record["update"], record["environment_steps"], record.get("elapsed_seconds", ""),
                deterministic["total"]["episodes"], "self_play" if self_play else "scripted",
                record.get("latest_checkpoint_update", ""), record.get("best_older_checkpoint_update", ""),
                *_summary(deterministic), *_summary(stochastic), guard,
            ])
            for policy, evaluation in (("deterministic", deterministic), ("stochastic", stochastic)):
                for side, key in (("total", "total"), ("p1", "as_p1"), ("p2", "as_p2")):
                    for style in evaluation[key].get("styles", []):
                        style_rows.append([
                            record["update"], policy, side, style["style"], style["episodes"],
                            style["wins"], style["losses"], style["draws"], style["win_rate"],
                        ])
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, header, rows in (("evaluations.csv", SUMMARY_HEADER, summary_rows),
                               ("evaluation_styles.csv", STYLE_HEADER, style_rows)):
        with (output_directory / name).open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    return len(summary_rows), len(style_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to the run directory itself.")
    args = parser.parse_args()
    evaluations, styles = export(args.run_directory, args.output_dir or args.run_directory)
    print(f"wrote {evaluations} evaluation rows and {styles} style rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

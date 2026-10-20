"""Statistical report and plot dashboard for training runs.

    python tools/run_report.py runs/<run>                 # report.json + dashboard.html in the run
    python tools/run_report.py runs/a runs/b runs/c ...   # also an across-seed summary
    python tools/run_report.py runs/<run> --stdout        # print the JSON report

Per run it reports binomial (Wilson) confidence intervals for the held-out win
rate, a draw-aware score (win = 1, draw = 0.5), side robustness (P1 minus P2
with a Newcombe interval), style robustness (pooled per-style win rates over
the final window with intervals, and the worst style's lower bound), and the
normalized area under the evaluation curve (AUC). Across runs it adds
Student-t intervals for final score and AUC.

dashboard.html is self-contained (inline SVG, no dependencies) and refreshes
itself, so it can be left open during training; scripts/training.py status
regenerates it.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_seed_matrix import across  # noqa: E402

Z95 = 1.959963984540054
FINAL_WINDOW = 10


def wilson(successes: float, trials: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval; draws may be counted as half successes."""

    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_difference(s1: float, n1: float, s2: float, n2: float) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid score interval."""

    p1, p2 = s1 / n1, s2 / n2
    l1, u1 = wilson(s1, n1)
    l2, u2 = wilson(s2, n2)
    difference = p1 - p2
    low = difference - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = difference + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return difference, max(-1.0, low), min(1.0, high)


def normalized_auc(updates: list[int], values: list[float]) -> float | None:
    if len(updates) < 2 or updates[-1] == updates[0]:
        return None
    area = sum((updates[i + 1] - updates[i]) * (values[i] + values[i + 1]) / 2 for i in range(len(updates) - 1))
    return area / (updates[-1] - updates[0])


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a line being written right now
    return rows


def run_report(run: Path) -> dict:
    evaluations = read_csv(run / "evaluations.csv")
    styles = read_csv(run / "evaluation_styles.csv")
    if not evaluations:
        return {"run_directory": str(run), "evaluations": 0}
    for row in evaluations:
        for key in ("update", "episodes"):
            row[key] = int(float(row[key]))
        for key in ("win_rate", "p1_win_rate", "p2_win_rate", "draw_rate", "worst_style_win_rate",
                    "stochastic_win_rate"):
            row[key] = float(row[key])
    latest = evaluations[-1]
    episodes = latest["episodes"]
    wins = latest["win_rate"] * episodes
    draws = latest["draw_rate"] * episodes
    side_episodes = episodes / 2  # held-out evaluation is side balanced
    side = newcombe_difference(latest["p1_win_rate"] * side_episodes, side_episodes,
                               latest["p2_win_rate"] * side_episodes, side_episodes)
    window = evaluations[-FINAL_WINDOW:]
    pooled_trials = sum(row["episodes"] for row in window)
    pooled_wins = sum(row["win_rate"] * row["episodes"] for row in window)
    report = {
        "run_directory": str(run),
        "evaluations": len(evaluations),
        "last_update": latest["update"],
        "latest": {
            "update": latest["update"], "episodes": episodes,
            "win_rate": latest["win_rate"], "win_rate_ci95": wilson(wins, episodes),
            "score": (wins + 0.5 * draws) / episodes, "score_ci95": wilson(wins + 0.5 * draws, episodes),
            "side_gap_p1_minus_p2": side[0], "side_gap_ci95": side[1:],
            "side_gap_significant": side[1] > 0 or side[2] < 0,
        },
        "final_window": {
            "evaluations": len(window), "episodes": pooled_trials,
            "win_rate": pooled_wins / pooled_trials, "win_rate_ci95": wilson(pooled_wins, pooled_trials),
        },
        "auc_win_rate": normalized_auc([row["update"] for row in evaluations],
                                       [row["win_rate"] for row in evaluations]),
        "auc_stochastic_win_rate": normalized_auc([row["update"] for row in evaluations],
                                                  [row["stochastic_win_rate"] for row in evaluations]),
    }
    window_updates = {row["update"] for row in window}
    pooled: dict[str, list[float]] = {}
    for row in styles:
        if (int(row["update"]) in window_updates and row["policy"] == "deterministic" and row["side"] == "total"):
            entry = pooled.setdefault(row["style"], [0.0, 0.0, 0.0])
            entry[0] += float(row["wins"])
            entry[1] += float(row["draws"])
            entry[2] += float(row["episodes"])
    if pooled:
        per_style = {style: {"episodes": int(n), "win_rate": w / n, "win_rate_ci95": wilson(w, n),
                             "score": (w + 0.5 * d) / n}
                     for style, (w, d, n) in sorted(pooled.items(), key=lambda item: int(item[0])) if n}
        worst = min(per_style, key=lambda style: per_style[style]["win_rate"])
        report["styles_final_window"] = per_style
        report["worst_style"] = {"style": worst, **per_style[worst]}
    return report


def seed_summary(reports: list[dict]) -> dict:
    usable = [report for report in reports if report.get("evaluations")]
    summary: dict = {"runs": len(reports), "runs_with_evaluations": len(usable)}
    for key, getter in (("final_window_win_rate", lambda r: r["final_window"]["win_rate"]),
                        ("auc_win_rate", lambda r: r["auc_win_rate"]),
                        ("latest_side_gap", lambda r: r["latest"]["side_gap_p1_minus_p2"])):
        values = [getter(report) for report in usable if getter(report) is not None]
        if values:
            summary[key] = across(values)
    return summary


# ---------------------------------------------------------------- dashboard

def _svg_chart(title: str, series: list[tuple[str, list[float], list[float], str]],
               y_range: tuple[float, float] | None = None, width: int = 560, height: int = 220) -> str:
    points = [(x, y) for _, xs, ys, _ in series for x, y in zip(xs, ys) if math.isfinite(y)]
    if not points:
        return f'<figure><figcaption>{html.escape(title)}</figcaption><p class="empty">no data yet</p></figure>'
    x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
    y0, y1 = y_range or (min(p[1] for p in points), max(p[1] for p in points))
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    left, right, top, bottom = 48, 8, 10, 22
    sx = lambda x: left + (x - x0) / (x1 - x0) * (width - left - right)
    sy = lambda y: top + (1 - (y - y0) / (y1 - y0)) * (height - top - bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
             f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>']
    for value in (y0, (y0 + y1) / 2, y1):
        parts.append(f'<text class="tick" x="{left - 4}" y="{sy(value) + 4:.1f}" text-anchor="end">{value:.3g}</text>')
    parts.append(f'<text class="tick" x="{left}" y="{height - 6}">{x0:,.0f}</text>')
    parts.append(f'<text class="tick" x="{width - right}" y="{height - 6}" text-anchor="end">{x1:,.0f}</text>')
    legend = []
    for name, xs, ys, css in series:
        coordinates = " ".join(f"{sx(x):.1f},{sy(min(max(y, y0), y1)):.1f}" for x, y in zip(xs, ys) if math.isfinite(y))
        if coordinates:
            parts.append(f'<polyline class="{css}" points="{coordinates}"/>')
            legend.append(f'<span class="key {css}">{html.escape(name)}</span>')
    parts.append("</svg>")
    return (f'<figure><figcaption>{html.escape(title)} {" ".join(legend)}</figcaption>'
            + "".join(parts) + "</figure>")


def dashboard_html(run: Path, report: dict) -> str:
    metrics = read_metrics(run / "metrics.jsonl")
    evaluations = read_csv(run / "evaluations.csv")
    updates = [float(row["update"]) for row in metrics]

    def metric(key: str) -> list[float]:
        return [float(row.get(key, math.nan)) if row.get(key) is not None else math.nan for row in metrics]

    throughput_x, throughput_y = [], []
    for previous, current in zip(metrics, metrics[1:]):
        elapsed = current["elapsed_seconds"] - previous["elapsed_seconds"]
        if elapsed > 0:
            throughput_x.append(float(current["update"]))
            throughput_y.append((current["environment_steps"] - previous["environment_steps"]) / elapsed / 1e6)
    eval_x = [float(row["update"]) for row in evaluations]

    def column(key: str) -> list[float]:
        return [float(row[key]) for row in evaluations]

    gap = [abs(float(row["p1_win_rate"]) - float(row["p2_win_rate"])) for row in evaluations]
    charts = [
        _svg_chart("Held-out win rate", [
            ("deterministic", eval_x, column("win_rate") if evaluations else [], "s1"),
            ("stochastic", eval_x, column("stochastic_win_rate") if evaluations else [], "s2"),
            ("worst style", eval_x, column("worst_style_win_rate") if evaluations else [], "s3")], (0.0, 1.0)),
        _svg_chart("Side gap |P1 - P2|", [("side gap", eval_x, gap, "s1")], (0.0, max([0.2] + gap))),
        _svg_chart("Rollout reward (mean per sample)", [("reward", updates, metric("rollout_reward_mean"), "s1")]),
        _svg_chart("Entropy (nats)", [("decision", updates, metric("decision_entropy"), "s1"),
                                      ("rollout decision", updates, metric("rollout_decision_entropy"), "s2"),
                                      ("all samples", updates, metric("entropy"), "s3")]),
        _svg_chart("Approximate KL and clip fraction", [("KL", updates, metric("approximate_kl"), "s1"),
                                                         ("clip fraction", updates, metric("clip_fraction"), "s2")]),
        _svg_chart("Throughput (M environment decisions/s)", [("throughput", throughput_x, throughput_y, "s1")]),
        _svg_chart("Value loss", [("value loss", updates, metric("value_loss"), "s1")]),
    ]
    latest = report.get("latest") or {}
    summary = "no evaluations yet"
    if latest:
        low, high = latest["win_rate_ci95"]
        g_low, g_high = latest["side_gap_ci95"]
        summary = (f"update {latest['update']:,}: win rate {latest['win_rate']:.1%} "
                   f"(95% CI {low:.1%} to {high:.1%}, {latest['episodes']} episodes); side gap "
                   f"{latest['side_gap_p1_minus_p2']:+.1%} (CI {g_low:+.1%} to {g_high:+.1%}); "
                   f"AUC {report['auc_win_rate']:.3f}" if report.get("auc_win_rate") is not None else "")
        worst = report.get("worst_style")
        if worst:
            summary += (f"; worst style {worst['style']} at {worst['win_rate']:.1%} "
                        f"(CI low {worst['win_rate_ci95'][0]:.1%})")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Run dashboard</title>
<style>
:root {{ --bg:#fff; --fg:#1d232b; --muted:#66707c; --grid:#d5dae0; --c1:#2f6fd6; --c2:#d9822b; --c3:#2e9c6a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15191e; --fg:#e4e8ec; --muted:#98a2ad; --grid:#38414b;
  --c1:#6ea2ff; --c2:#f0a35c; --c3:#5cc795; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; margin:0 16px 32px; }}
h1 {{ font-size:18px; margin:18px 0 4px; }} p.summary {{ color:var(--muted); margin:0 0 16px; }}
main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr)); gap:16px; }}
figure {{ margin:0; }} figcaption {{ font-weight:600; margin-bottom:4px; }}
svg {{ width:100%; height:auto; }} polyline {{ fill:none; stroke-width:1.6; }}
.axis {{ stroke:var(--grid); }} .tick {{ fill:var(--muted); font-size:11px; }} .empty {{ color:var(--muted); }}
polyline.s1 {{ stroke:var(--c1); }} polyline.s2 {{ stroke:var(--c2); }} polyline.s3 {{ stroke:var(--c3); }}
.key {{ font-weight:400; font-size:12px; margin-left:8px; }} .key::before {{ content:"\\25A0 "; }}
.key.s1::before {{ color:var(--c1); }} .key.s2::before {{ color:var(--c2); }} .key.s3::before {{ color:var(--c3); }}
</style></head><body>
<h1>{html.escape(run.name)}</h1><p class="summary">{html.escape(summary)}</p>
<main>{''.join(charts)}</main></body></html>
"""


def write_artifacts(run: Path) -> dict:
    report = run_report(run)
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary = run / "dashboard.html.tmp"
    temporary.write_text(dashboard_html(run, report), encoding="utf-8")
    temporary.replace(run / "dashboard.html")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--stdout", action="store_true", help="Print the report instead of only writing files.")
    args = parser.parse_args()
    reports = [write_artifacts(run) for run in args.runs]
    result = reports[0] if len(reports) == 1 else {"runs": reports, "across_seeds": seed_summary(reports)}
    if args.stdout or len(reports) > 1:
        print(json.dumps(result, indent=2))
    else:
        print(f"wrote {args.runs[0] / 'report.json'} and {args.runs[0] / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

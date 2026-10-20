"""Reproducible V3 acceptance report.

    python tools/acceptance_report.py --seed-runs runs/seed_matrix_2027 runs/seed_matrix_2028 ... \\
        --replays artifacts/live_replays/*.jsonl --output docs/acceptance/v3_acceptance

Checks each V3 acceptance criterion against the repository's own evidence
(compiled catalog, character gates, validation ledgers, seed-matrix runs, live
replay logs) and writes <output>.json and <output>.md. Nothing is inferred: a
criterion without evidence fails. The report records the commit and the
catalog hash so it can be regenerated and compared. Exit code 0 only when
every criterion passes.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from aggregate_seed_matrix import MINIMUM_LONG_UPDATES, MINIMUM_SEEDS, load, per_seed  # noqa: E402
from t8_agent.live.feedback import compare_replay  # noqa: E402
from t8_agent.moves.catalog import compile_catalog, load_compiled_catalog  # noqa: E402
from t8_agent.moves.validation import evaluate_character_gate  # noqa: E402

SIDE_GAP_LIMIT = 0.10
WORST_STYLE_FLOOR = 0.50


def commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def criterion(name: str, passed: bool, evidence: str) -> dict:
    return {"criterion": name, "pass": bool(passed), "evidence": evidence}


def build_report(data_root: Path, seed_runs: list[Path], replays: list[Path]) -> dict:
    catalog_path = data_root / "generated" / "full_move_catalog.json"
    committed = load_compiled_catalog(catalog_path)
    rebuilt = compile_catalog(data_root)
    gates = [evaluate_character_gate(committed, row["slug"], data_root / "validation")
             for row in committed.characters]
    ready = [gate.character for gate in gates if gate.ready]
    parser_pending = sum(gate.parser_pending for gate in gates)
    conflicts = sum(gate.source_conflicts for gate in gates)
    measured = sum(gate.move_count - gate.measurement_pending for gate in gates)
    practiced = sum(gate.move_count - gate.practice_pending for gate in gates)
    total_moves = sum(gate.move_count for gate in gates)
    criteria = [
        criterion("Generated catalog is reproducible from its sources",
                  rebuilt.catalog_sha256 == committed.catalog_sha256,
                  f"committed {committed.catalog_sha256[:12]}, rebuilt {rebuilt.catalog_sha256[:12]}"),
        criterion("Every documented command is parsed", parser_pending == 0,
                  f"{parser_pending} of {total_moves} commands need parser review"),
        criterion("No inconsistent source frame records", conflicts == 0, f"{conflicts} inconsistent records"),
        criterion("At least one fighter passes the complete character gate", bool(ready),
                  f"ready: {', '.join(ready) or 'none'} of {len(gates)} roster records"),
        criterion("Measured combat fields with provenance", measured == total_moves,
                  f"{measured} of {total_moves} moves fully measured"),
        criterion("Offline Practice validation with provenance", practiced == total_moves,
                  f"{practiced} of {total_moves} moves passed"),
    ]

    seeds = [per_seed(run, rows) for run in seed_runs if (rows := load(run))]
    long_seeds = [seed for seed in seeds if seed["last_update"] >= MINIMUM_LONG_UPDATES]
    criteria.append(criterion(
        f"At least {MINIMUM_SEEDS} long training seeds (>= {MINIMUM_LONG_UPDATES} updates)",
        len(long_seeds) >= MINIMUM_SEEDS,
        f"{len(long_seeds)} long of {len(seed_runs)} given runs"))
    if long_seeds:
        worst_gap = max(seed["final_side_gap"] for seed in long_seeds)
        worst_style = min(seed["final_worst_style_win_rate"] for seed in long_seeds)
        criteria.append(criterion(f"Side gap at most {SIDE_GAP_LIMIT:.2f} on every long seed",
                                  worst_gap <= SIDE_GAP_LIMIT, f"largest final side gap {worst_gap:.3f}"))
        criteria.append(criterion(f"Worst held-out style at least {WORST_STYLE_FLOOR:.2f} on every long seed",
                                  worst_style >= WORST_STYLE_FLOOR, f"lowest final worst-style {worst_style:.3f}"))
    else:
        criteria.append(criterion(f"Side gap at most {SIDE_GAP_LIMIT:.2f} on every long seed", False,
                                  "no long seed runs"))
        criteria.append(criterion(f"Worst held-out style at least {WORST_STYLE_FLOOR:.2f} on every long seed",
                                  False, "no long seed runs"))

    comparisons = [compare_replay(path) for path in replays]
    sent = sum(item["sent"] for item in comparisons)
    mismatches = sum(len(item["input_mismatches"]) for item in comparisons)
    criteria.append(criterion("Live replays: every sent input matches the planned controller sequence",
                              bool(comparisons) and sent > 0 and mismatches == 0,
                              f"{len(comparisons)} replays, {sent} sent decisions, {mismatches} mismatches"))

    # The CUDA simulator runs the scalar engine's own frame step (full_combat_core.hpp); the
    # parity test replays randomized CPU fights on the GPU. Its last result comes from ctest.
    parity_log = REPO_ROOT / "build-gpu" / "Testing" / "Temporary" / "LastTest.log"
    parity_text = parity_log.read_text(encoding="utf-8", errors="replace") if parity_log.exists() else ""
    parity_passed = "full-move GPU tests passed" in parity_text and "0 divergent" in parity_text
    criteria.append(criterion("Full-move engine mirrored in CUDA with randomized parity", parity_passed,
                              "t8_v2_full_move_gpu passed with 0 divergent fights in the last ctest run"
                              if parity_passed else "run ctest -R t8_v2_full_move_gpu (no passing result recorded)"))
    return {
        "generated_on": date.today().isoformat(),
        "commit": commit(),
        "catalog_sha256": committed.catalog_sha256,
        "accepted": all(item["pass"] for item in criteria),
        "criteria": criteria,
        "characters": [gate.to_dict() for gate in gates],
        "seeds": seeds,
        "replays": [{"path": str(path), **summary} for path, summary in zip(replays, comparisons)],
    }


def markdown(report: dict) -> str:
    lines = [f"# V3 acceptance report ({report['generated_on']})", "",
             f"Commit `{report['commit']}`, catalog `{report['catalog_sha256'][:16]}`. "
             f"Accepted: **{'yes' if report['accepted'] else 'no'}** "
             f"({sum(item['pass'] for item in report['criteria'])} of {len(report['criteria'])} criteria pass).",
             "", "| Criterion | Result | Evidence |", "|---|---|---|"]
    for item in report["criteria"]:
        lines.append(f"| {item['criterion']} | {'pass' if item['pass'] else 'FAIL'} | {item['evidence']} |")
    closest = sorted(report["characters"], key=lambda gate: (
        gate["move_count"] == 0,
        gate["parser_pending"] + gate["source_conflicts"] + gate["measurement_pending"] + gate["practice_pending"]
        + len(gate["scenario_pending"]) + len(gate["route_pending"])))[:5]
    lines += ["", "Closest characters to their gate:", ""]
    lines += [f"- {gate['character']}: " + ("; ".join(gate["blockers"]) or "ready") for gate in closest]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--seed-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--replays", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "acceptance" / "v3_acceptance")
    args = parser.parse_args()
    report = build_report(args.data_root, args.seed_runs, args.replays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

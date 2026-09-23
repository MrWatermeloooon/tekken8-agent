"""Exports per-move engine bindings for the scalar full-combat engine.

Joins, per move:
  * the compiled catalog (parsed command, hit levels, damage, frame data,
    mechanics, stances, situational requirements);
  * the raw source row (frame-advantage suffixes: a = launch, d = knockdown;
    frame windows written in the notes: low/high crush, power crush, parry
    state, floating state, intangibility);
  * measured geometry from data/measurements/<slug>.yaml, if any;
  * Practice validation status from data/validation/<slug>.json.

The C++ binder (include/t8_v2/full_combat_binding.hpp) turns rows into
FullMoveSpec values and refuses anything unvalidated, unmeasured, or reactive.
Nothing is estimated here: missing values stay empty.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from t8_agent.moves.catalog import load_compiled_catalog  # noqa: E402
from t8_agent.moves.validation import load_validation_ledger  # noqa: E402

MEASUREMENT_KEYS = ("active_frames", "range", "tracking_left", "tracking_right", "pushback", "travel", "hit_frames")
WINDOW_PATTERNS = {
    "low_crush": re.compile(r"low crush\s*~?(\d+)~(\d*)", re.IGNORECASE),
    "high_crush": re.compile(r"high crush\s*(\d+)~(\d*)", re.IGNORECASE),
    "power_crush": re.compile(r"power crush\s*(\d+)~(\d*)", re.IGNORECASE),
    "parry": re.compile(r"parry state\s*(\d+)~(\d*)", re.IGNORECASE),
    "airborne": re.compile(r"floating state\s*(\d+)~(\d*)", re.IGNORECASE),
    "invincible": re.compile(r"(?:^|[\s;])is(\d+)~(\d*)", re.IGNORECASE),
}
OPEN_WINDOW_END = 999  # "N~" in the notes: until the move ends
EFFECTS = {"a": "launch", "d": "knockdown"}
FIELDS = [
    "stable_id", "character", "name", "command", "parser_status", "source_consistency", "reactive",
    "hit_levels", "damages", "startup", "recovery", "block_advantage", "hit_advantage", "hit_effect",
    "counter_hit_advantage", "counter_hit_effect", "frame_data_variable",
    "homing", "tornado", "wall_break", "floor_break", "balcony_break", "heat_engager", "requires_heat",
    "requires_rage", "rage_art", "posture", "stances", "situations", "result_stance", "automatic_transition",
    "low_crush", "high_crush", "power_crush", "parry", "airborne", "invincible",
    "practice_status",
] + [f"measured_{key}" for key in MEASUREMENT_KEYS]


def parse_advantage(text: object) -> tuple[int | None, str, bool]:
    """'+21a~+64a (-5~+38)' -> (21, 'launch', variable=True). Empty -> (None, 'stun', False)."""
    main = str(text or "").split("(")[0].strip()
    values = re.findall(r"([+-]?\d+)([a-z]*)", main)
    if not values:
        return None, "stun", False
    number, suffix = values[0]
    effect = EFFECTS.get(suffix[:1], "stun")
    variable = len(values) > 1 or "~" in main
    return int(number), effect, variable


def parse_windows(notes: str) -> dict[str, str]:
    """All spans of each window kind, e.g. {"airborne": "5~13|34~36"}."""
    windows = {}
    for name, pattern in WINDOW_PATTERNS.items():
        spans = [f"{int(first)}~{int(last or OPEN_WINDOW_END)}" for first, last in pattern.findall(notes)]
        if spans:
            windows[name] = "|".join(spans)
    return windows


def automatic_transitions(notes: str) -> str:
    """Transitions that happen without an extra input ("Transitions to IZU on hit or block").

    Optional branches ("Enter GEN ... with F") are separate move variants and
    are not listed; the base move binds without them.
    """
    lines = [line.strip("* ").strip() for line in notes.splitlines()]
    automatic = [line for line in lines
                 if re.match(r"(transition|enter)", line, re.IGNORECASE)
                 and not re.search(r"\bwi+th\b", line)]  # "wiith": typo in the source
    return " / ".join(automatic)


def load_measurements(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(key): dict(value or {}) for key, value in (document.get("moves") or {}).items()}


def join(values) -> str:
    return "|".join(str(value) for value in values)


def one_line(value: object) -> str:
    """Collapses whitespace so every CSV record is one line (source names can hold HTML lists)."""
    return " ".join(str(value).split())


def export(catalog_path: Path, data_root: Path, output: Path, characters: list[str] | None = None) -> int:
    catalog = load_compiled_catalog(catalog_path)
    slugs = {row["id"]: row["slug"] for row in catalog.characters}
    rows = []
    source_cache: dict[str, dict[str, dict]] = {}
    for move in catalog.moves:
        slug = slugs[move["character_id"]]
        if characters and slug not in characters:
            continue
        if slug not in source_cache:
            source = yaml.safe_load((data_root / "characters" / f"{slug}.yaml").read_text(encoding="utf-8")) or {}
            source_cache[slug] = {
                "__rows__": {str(row.get("source_id")): row for row in source.get("moves") or []},
                "__measure__": load_measurements(data_root / "measurements" / f"{slug}.yaml"),
                "__ledger__": (load_validation_ledger(data_root / "validation" / f"{slug}.json").get("moves") or {}),
            }
        cache = source_cache[slug]
        raw = cache["__rows__"].get(move["source_id"], {})
        corrected = {**raw, **((move.get("corrections") or {}).get("fields") or {})}
        block, _, block_variable = parse_advantage(corrected.get("block"))
        hit, hit_effect, hit_variable = parse_advantage(corrected.get("hit"))
        counter, counter_effect, counter_variable = parse_advantage(corrected.get("counter_hit"))
        startup = move["startup"]
        recovery = move["recovery"]
        mechanics = move["mechanics"]
        measured = cache["__measure__"].get(move["stable_id"], {})
        steps = move["command"]["steps"]
        row = {
            "stable_id": move["stable_id"], "character": slug, "name": one_line(move["name"]),
            "command": one_line(move["command"]["raw"]), "parser_status": move["command"]["parser_status"],
            "source_consistency": move["validation"]["source_consistency"],
            "reactive": int(any(str(step.get("requirement") or "").startswith("PARRY_SUCCESS") for step in steps)),
            "hit_levels": join(move["hit_levels"]), "damages": join(move["damage"]),
            "startup": "" if startup is None else startup["min"],
            "recovery": "" if recovery is None else recovery["min"],
            "block_advantage": "" if block is None else block,
            "hit_advantage": "" if hit is None else hit, "hit_effect": hit_effect,
            "counter_hit_advantage": "" if counter is None else counter,
            "counter_hit_effect": counter_effect if counter is not None else "",
            "frame_data_variable": int(block_variable or hit_variable or counter_variable or
                                       (startup is not None and startup["min"] != startup["max"])),
            "homing": int(mechanics["homing"]), "tornado": int(mechanics["tornado"]),
            "wall_break": int(mechanics["wall_break"]), "floor_break": int(mechanics["floor_break"]),
            "balcony_break": int(mechanics["balcony_break"]), "heat_engager": int(mechanics["heat_engager"]),
            "requires_heat": int(mechanics["requires_heat"]), "requires_rage": int(mechanics["requires_rage"]),
            "rage_art": int(mechanics["rage_art"]),
            "posture": move["legal_state"]["posture"], "stances": join(move["legal_state"]["stances"]),
            "situations": join(move["legal_state"].get("situations", [])),
            "result_stance": mechanics.get("stance_transition") or "",
            "automatic_transition": automatic_transitions(str(raw.get("notes") or "")),
            "practice_status": (cache["__ledger__"].get(move["stable_id"]) or {}).get("status", "pending"),
            **{name: "" for name in WINDOW_PATTERNS},
            **parse_windows(str(raw.get("notes") or "")),
        }
        for key in MEASUREMENT_KEYS:
            value = measured.get(key)
            row[f"measured_{key}"] = join(value) if isinstance(value, list) else ("" if value is None else value)
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data/generated/full_move_catalog.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/generated/full_combat_bindings.csv")
    parser.add_argument("--character", action="append", help="Limit to these slugs (default: all).")
    args = parser.parse_args()
    count = export(args.catalog, args.data_root, args.output, args.character)
    print(json.dumps({"rows": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

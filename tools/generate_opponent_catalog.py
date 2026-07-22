from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from t8_agent.roster.catalog import OpponentProfile, load_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the 50-profile-per-character roster catalog.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()
    catalog = load_catalog(args.data_root)
    generated = args.data_root / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    csv_path = generated / "opponent_profiles.csv"
    _write_profiles(csv_path, catalog.profiles)
    move_path = generated / "character_move_specs.csv"
    move_rows: list[dict[str, Any]] = []
    module_root = args.data_root / "character_modules"
    available_frame_data = 0
    for character in catalog.characters:
        source = args.data_root / "characters" / f"{character.slug}.yaml"
        knowledge = _build_matchup_module(character, source)
        move_rows.extend(_distill_character_moves(character, source))
        output = module_root / character.slug / "matchup.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(knowledge, sort_keys=False, allow_unicode=False), encoding="utf-8")
        available_frame_data += int(knowledge["frame_data_status"] == "available")
    _write_rows(move_path, move_rows)
    digest = sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "roster_as_of": catalog.as_of,
        "roster_source": catalog.source,
        "characters": len(catalog.characters),
        "archetypes": len(catalog.archetypes),
        "variations_per_archetype": len(catalog.variations),
        "profiles_per_character": len(catalog.archetypes) * len(catalog.variations),
        "total_profiles": len(catalog.profiles),
        "frame_data_characters": available_frame_data,
        "announced_unreleased": list(catalog.announced_unreleased),
        "profiles_sha256": digest,
        "character_move_specs": len(move_rows),
        "character_move_specs_sha256": sha256(move_path.read_bytes()).hexdigest(),
    }
    (generated / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _write_profiles(path: Path, profiles: tuple[OpponentProfile, ...]) -> None:
    rows = [asdict(profile) for profile in profiles]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _distill_character_moves(character: Any, frame_path: Path) -> list[dict[str, Any]]:
    data: dict[str, Any] = {}
    if frame_path.exists():
        loaded = yaml.safe_load(frame_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    moves = list(data.get("moves") or [])
    roles = [
        ("jab", "1", "high", 10, 7.0, 0.82, 13, 0.08, False),
        ("df1", "df+1", "mid", 13, 12.0, 0.95, 18, 0.11, False),
        ("f2", "f+2", "mid", 16, 18.0, 1.25, 22, 0.16, False),
        ("db3", "db+3", "low", 18, 10.0, 0.90, 24, 0.06, False),
        ("hopkick", "uf+4", "mid", 15, 21.0, 0.88, 30, 0.18, True),
        ("throw", None, "throw", 12, 25.0, 0.48, 24, 0.22, False),
    ]
    rows: list[dict[str, Any]] = []
    for slot, command, fallback_level, fallback_startup, fallback_damage, move_range, recovery, pushback, launches in roles:
        selected = _select_role_move(moves, command, slot)
        startup = _first_integer(selected.get("startup")) if selected else None
        damage = _damage_total(selected.get("damage")) if selected else None
        block = _block_frames(selected.get("block")) if selected else None
        hit = _signed_frames(selected.get("hit")) if selected else None
        selected_level = _first_hit_level(selected.get("hit_level")) if selected else "none"
        hit_level = selected_level if selected_level in {"high", "mid", "low", "throw"} else fallback_level
        actual_recovery = recovery + (4 if selected and "power_crush" in selected.get("tags", []) else 0)
        rows.append({
            "character_id": character.id,
            "character": character.slug,
            "slot": slot,
            "source_status": "frame_data" if selected else "abstract_fallback",
            "source_move_id": "" if selected is None else selected.get("id", ""),
            "source_command": "" if selected is None else selected.get("command", ""),
            "hit_level": hit_level,
            "startup": fallback_startup if startup is None else max(1, startup),
            "active": 2 if slot in {"jab", "throw"} else (4 if slot == "hopkick" else 3),
            "recovery": actual_recovery,
            "damage": fallback_damage if damage is None or damage <= 0 else min(45.0, damage),
            "range": move_range,
            "hitstun": max(4, actual_recovery + (6 if hit is None else hit)),
            "blockstun": max(0, actual_recovery + (-8 if block is None else block)),
            "pushback": pushback,
            "whiff_recovery": 4 if slot in {"f2", "hopkick"} else 0,
            "launches": int(launches or (selected is not None and "launcher" in selected.get("tags", []))),
        })
    return rows


def _select_role_move(moves: list[dict[str, Any]], command: str | None, slot: str) -> dict[str, Any] | None:
    if command is not None:
        for move in moves:
            if str(move.get("command", "")).strip().lower() == command.lower():
                return move
    if slot == "throw":
        candidates = [move for move in moves if _is_throw_move(move)]
    elif slot == "hopkick":
        candidates = [move for move in moves if "launcher" in move.get("tags", []) and "mid" in move.get("tags", [])]
    elif slot == "db3":
        candidates = [move for move in moves if "low" in move.get("tags", [])]
    elif slot == "jab":
        candidates = [move for move in moves if _first_hit_level(move.get("hit_level")) == "high"]
    else:
        candidates = [move for move in moves if _first_hit_level(move.get("hit_level")) == "mid"]
    if slot == "f2" and candidates:
        candidates.sort(
            key=lambda move: (
                (_damage_total(move.get("damage")) or 0.0)
                - max(0, (_first_integer(move.get("startup")) or 40) - 24) * 2.0
            ),
            reverse=True,
        )
        return candidates[0]
    candidates.sort(key=lambda move: _first_integer(move.get("startup")) or 999)
    return candidates[0] if candidates else None


def _first_integer(value: Any) -> int | None:
    match = re.search(r"(?<!\d)(\d+)", str(value or ""))
    return None if match is None else int(match.group(1))


def _signed_frames(value: Any) -> int | None:
    match = re.search(r"(?<!\d)([+-]\d+)", str(value or ""))
    return None if match is None else int(match.group(1))


def _damage_total(value: Any) -> float | None:
    values = [int(number) for number in re.findall(r"\d+", str(value or ""))]
    if not values:
        return None
    return float(sum(values[:6]))


def _first_hit_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("t") or "throw" in text: return "throw"
    if text.startswith("l"): return "low"
    if text.startswith("m"): return "mid"
    if text.startswith("h"): return "high"
    return "none"


def _is_throw_move(move: dict[str, Any]) -> bool:
    levels = [value.strip().lower() for value in re.split(r"[, ]+", str(move.get("hit_level", ""))) if value.strip()]
    command = str(move.get("command", ""))
    return any(value.startswith("t") for value in levels) and "," not in command


def _build_matchup_module(character: Any, frame_path: Path) -> dict[str, Any]:
    frame_data: dict[str, Any] = {}
    if frame_path.exists():
        loaded = yaml.safe_load(frame_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            frame_data = loaded
    moves = list(frame_data.get("moves") or [])
    available = frame_data.get("data_status", "available" if moves else "unavailable") == "available"
    punishable = sorted(
        (move for move in moves if (_block_frames(move.get("block")) or 0) <= -10),
        key=lambda move: _block_frames(move.get("block")) or 0,
    )
    launch_punishable = [move for move in punishable if (_block_frames(move.get("block")) or 0) <= -15]
    strings = [move for move in moves if "," in str(move.get("command", ""))]
    duckable = [move for move in strings if _contains_high(move) and "h" in str(move.get("hit_level", "")).lower()]
    stance_moves = [move for move in moves if "." in str(move.get("command", "")) or "stance_transition" in move.get("tags", [])]
    throws = [move for move in moves if "throw" in move.get("tags", [])]
    lows = [move for move in moves if "low" in move.get("tags", [])]
    power_crushes = [move for move in moves if "power_crush" in move.get("tags", [])]
    heat_threats = [move for move in moves if "heat" in move.get("tags", [])]
    long_range = [
        move for move in moves
        if "f,F" in str(move.get("command", ""))
        or "running" in str(move.get("name", "")).lower()
        or "long range" in str(move.get("notes", "")).lower()
    ]
    return {
        "schema_version": 1,
        "character_id": character.id,
        "character": character.slug,
        "display_name": character.name,
        "groups": list(character.groups),
        "specialty": character.specialty,
        "sidestep_tendency": character.sidestep,
        "signature_mechanics": list(character.signatures),
        "frame_data_status": "available" if available else "unavailable",
        "frame_data_source": frame_data.get("source_url"),
        "important_punishable_moves": _summaries(punishable, 32),
        "common_strings_and_gaps": _summaries(strings, 32),
        "duckable_highs": _summaries(duckable, 24),
        "stance_transitions": _summaries(stance_moves, 24),
        "throw_options": _summaries(throws, 24),
        "key_lows": _summaries(lows, 24),
        "power_crushes": _summaries(power_crushes, 16),
        "heat_threats": _summaries(heat_threats, 24),
        "long_range_attacks": _summaries(long_range, 16),
        "launch_punish_opportunities": _summaries(launch_punishable, 24),
        "jun_response_options": {
            "block_minus_10_to_12": "jab punish",
            "block_minus_13_to_14": "df1 or character-calibrated punish",
            "block_minus_15_or_worse": "hopkick launch punish",
            "low": "block low or low parry using timing history",
            "throw": "select 1, 2, or 1+2 break from throw history",
            "sidestep": character.sidestep,
        },
    }


def _summaries(moves: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    return [
        {
            "id": str(move.get("id", "")), "command": str(move.get("command", "")),
            "hit_level": str(move.get("hit_level", "")), "startup": str(move.get("startup", "")),
            "block": str(move.get("block", "")), "notes": str(move.get("notes", ""))[:240],
        }
        for move in moves[:limit]
    ]


def _block_frames(value: Any) -> int | None:
    text = str(value or "")
    match = re.search(r"(?<!\d)(-\d+)", text)
    return None if match is None else int(match.group(1))


def _contains_high(move: dict[str, Any]) -> bool:
    levels = [part.strip().lower() for part in re.split(r"[, ]+", str(move.get("hit_level", ""))) if part.strip()]
    return any(level.startswith("h") for level in levels[1:])


if __name__ == "__main__":
    raise SystemExit(main())

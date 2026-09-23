from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import yaml

from .notation import SITUATIONAL_REQUIREMENTS, parse_command


CATALOG_SCHEMA_VERSION = 2
ACTION_FEATURE_SIZE = 32


@dataclass(frozen=True)
class CompiledMoveCatalog:
    document: dict[str, Any]

    @property
    def characters(self) -> list[dict[str, Any]]:
        return self.document["characters"]

    @property
    def moves(self) -> list[dict[str, Any]]:
        return self.document["moves"]

    @property
    def catalog_sha256(self) -> str:
        return str(self.document["catalog_sha256"])


def compile_catalog(data_root: str | Path) -> CompiledMoveCatalog:
    root = Path(data_root)
    roster = _read_yaml(root / "roster.yaml")
    characters: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    stable_ids: set[str] = set()
    for roster_row in roster["characters"]:
        character_id = int(roster_row["id"])
        slug = str(roster_row["slug"])
        source_path = root / "characters" / f"{slug}.yaml"
        source = _read_yaml(source_path)
        source_moves = list(source.get("moves") or [])
        corrections = _load_corrections(root / "corrections" / f"{slug}.yaml", source_moves)
        stances = [str(value) for value in source.get("stances") or []]
        offset = len(moves)
        parsed_count = 0
        review_count = 0
        for local_id, row in enumerate(source_moves):
            correction = corrections.get(str(row.get("source_id") or ""))
            if correction:
                row = {**row, **correction["fields"]}
            compiled = _compile_move(character_id, slug, local_id, row, source, set(stances))
            if correction:
                compiled["validation"]["source"] = "imported+corrected"
                compiled["corrections"] = {key: correction[key] for key in
                                           ("fields", "reason", "evidence", "reviewer", "status")}
            stable_id = compiled["stable_id"]
            if stable_id in stable_ids:
                raise ValueError(f"duplicate stable move ID: {stable_id}")
            stable_ids.add(stable_id)
            parsed_count += compiled["command"]["parser_status"] == "parsed"
            review_count += compiled["command"]["parser_status"] != "parsed"
            moves.append(compiled)
        data_status = str(source.get("data_status", "unavailable"))
        characters.append({
            "id": character_id,
            "slug": slug,
            "display_name": str(roster_row["name"]),
            "data_status": data_status,
            "move_offset": offset,
            "move_count": len(source_moves),
            "parsed_move_count": parsed_count,
            "review_move_count": review_count,
            "stances": stances,
            "practice_validation": "pending" if source_moves else "blocked_missing_source",
            "training_enabled": False,
            "source_path": source_path.relative_to(root.parent).as_posix(),
            "source_url": str(source.get("source_url", "")),
            "retrieved_at": str(source.get("retrieved_at", "")),
            "source_sha256": str(source.get("source_sha256", "")),
        })

    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "roster_as_of": str(roster["as_of"]),
        "roster_source": str(roster["source"]),
        "character_count": len(characters),
        "move_count": len(moves),
        "action_feature_size": ACTION_FEATURE_SIZE,
        "characters": characters,
        "moves": moves,
        "announced_unreleased": list(roster.get("announced_unreleased") or []),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    payload["catalog_sha256"] = sha256(canonical).hexdigest()
    return CompiledMoveCatalog(payload)


def load_compiled_catalog(path: str | Path) -> CompiledMoveCatalog:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported full move catalog schema")
    expected = document.get("catalog_sha256")
    payload = dict(document)
    payload.pop("catalog_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if expected != sha256(canonical).hexdigest():
        raise ValueError("full move catalog checksum mismatch")
    if document.get("move_count") != len(document.get("moves", [])):
        raise ValueError("full move catalog count mismatch")
    return CompiledMoveCatalog(document)


_CORRECTABLE_FIELDS = {"command", "hit_level", "damage", "startup", "recovery", "block", "hit", "counter_hit"}


def _load_corrections(path: Path, source_moves: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reviewed corrections applied on top of an imported (checksum-pinned) source.

    Each entry names a source_id, the replaced fields, the reason, evidence
    URLs, and who reviewed it, so the imported file itself never changes.
    """
    if not path.exists():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {str(row.get("source_id")) for row in source_moves}
    result: dict[str, dict[str, Any]] = {}
    for entry in document.get("corrections") or []:
        source_id = str(entry.get("source_id", ""))
        fields = dict(entry.get("fields") or {})
        if source_id not in known:
            raise ValueError(f"correction for unknown source_id {source_id!r} in {path}")
        if source_id in result:
            raise ValueError(f"duplicate correction for {source_id!r} in {path}")
        if not fields or not set(fields) <= _CORRECTABLE_FIELDS:
            raise ValueError(f"correction for {source_id!r} must replace only {sorted(_CORRECTABLE_FIELDS)}")
        if not entry.get("reason") or not entry.get("evidence") or not entry.get("reviewer"):
            raise ValueError(f"correction for {source_id!r} needs reason, evidence, and reviewer")
        result[source_id] = {
            "fields": fields, "reason": str(entry["reason"]),
            "evidence": [str(value) for value in entry["evidence"]],
            "reviewer": str(entry["reviewer"]), "status": str(entry.get("status", "pending_practice")),
        }
    return result


def write_catalog(catalog: CompiledMoveCatalog, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog.document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _compile_move(character_id: int, slug: str, local_id: int, row: dict[str, Any],
                  source: dict[str, Any], stances: set[str]) -> dict[str, Any]:
    command_raw = str(row.get("command", "")).strip()
    command = parse_command(command_raw, stances)
    source_key = str(row.get("source_id") or row.get("id") or local_id)
    source_index = int(row["source_index"]) if row.get("source_index") is not None else local_id
    stable_id = f"{slug}:{source_index}"
    levels = _parse_hit_levels(row.get("hit_level"))
    damage = _parse_number_sequence(row.get("damage"))
    startup = _parse_range(row.get("startup"), "i")
    recovery = _parse_range(row.get("recovery"), "r")
    block = _parse_signed_range(row.get("block"))
    hit = _parse_signed_range(row.get("hit"))
    counter_hit = _parse_signed_range(row.get("counter_hit"))
    tags = sorted({str(value).lower() for value in (row.get("tags") or [])})
    notes = str(row.get("notes") or "")
    mechanics = _mechanics(tags, notes, command.requirements)
    estimates = {
        "active_frames": None,
        "range": None,
        "tracking_left": None,
        "tracking_right": None,
        "pushback": None,
        "collision": None,
    }
    source_issues: list[str] = []
    if startup is not None and startup["max"] < startup["min"]:
        source_issues.append("startup_range_descends")
    if recovery is not None and recovery["max"] < recovery["min"]:
        source_issues.append("recovery_range_descends")
    validation = {
        "source": "imported",
        "source_consistency": "blocked" if source_issues else "valid",
        "issues": source_issues,
        "parser": command.parser_status,
        "practice": "pending",
        "simulation": "blocked_missing_measurements" if any(value is None for value in estimates.values()) else "ready",
    }
    return {
        "character_id": character_id,
        "local_id": local_id,
        "stable_id": stable_id,
        "source_id": source_key,
        "source_index": source_index,
        "name": str(row.get("name") or ""),
        "command": command.to_dict(),
        "hit_levels": levels,
        "damage": damage,
        "startup": startup,
        "recovery": recovery,
        "block_advantage": block,
        "hit_advantage": hit,
        "counter_hit_advantage": counter_hit,
        "tags": tags,
        "mechanics": mechanics,
        "legal_state": _legal_state(command.requirements, mechanics),
        "measurements": estimates,
        "action_features": _action_features(local_id, levels, damage, startup, recovery, block, tags, mechanics),
        "notes": notes,
        "validation": validation,
        "provenance": {
            "source_url": str(source.get("source_url", "")),
            "retrieved_at": str(source.get("retrieved_at", "")),
            "source_sha256": str(source.get("source_sha256", "")),
            "source_row": source_index,
        },
    }


def _legal_state(requirements: tuple[str, ...], mechanics: dict[str, Any]) -> dict[str, Any]:
    requirement_set = {value.upper() for value in requirements}
    posture = "standing"
    if requirement_set & {"FC", "HFC"}:
        posture = "crouching"
    elif "WS" in requirement_set:
        posture = "while_standing"
    situations = sorted(requirement_set & SITUATIONAL_REQUIREMENTS)
    stance_requirements = sorted(requirement_set - {"H", "R", "FC", "HFC", "WS", "SS", "WR"} -
                                 SITUATIONAL_REQUIREMENTS)
    return {
        "posture": posture,
        "requires_heat": bool(mechanics["requires_heat"]),
        "requires_rage": bool(mechanics["requires_rage"]),
        "requires_sidestep": "SS" in requirement_set,
        "requires_running": "WR" in requirement_set,
        "stances": stance_requirements,
        "situations": situations,
        "source": "command_notation",
    }


def _parse_hit_levels(value: Any) -> list[str]:
    text = str(value or "").lower()
    aliases = {"h": "high", "m": "mid", "l": "low", "t": "throw",
               "sm": "special_mid", "sl": "special_low"}
    result: list[str] = []
    for token in re.split(r"[, ]+", text):
        cleaned = re.sub(r"[^a-z]", "", token)
        if not cleaned:
            continue
        result.append(aliases.get(cleaned, cleaned))
    return result


def _parse_number_sequence(value: Any) -> list[float]:
    return [float(match) for match in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]


def _parse_range(value: Any, prefix: str) -> dict[str, int] | None:
    text = str(value or "")
    match = re.search(rf"{prefix}\s*(\d+)(?:~(\d+))?", text, re.IGNORECASE)
    if not match:
        return None
    return {"min": int(match.group(1)), "max": int(match.group(2) or match.group(1))}


def _parse_signed_range(value: Any) -> dict[str, int] | None:
    values = [int(item) for item in re.findall(r"(?<!\d)[+-]\d+", str(value or ""))]
    return None if not values else {"min": min(values), "max": max(values)}


def _mechanics(tags: list[str], notes: str, requirements: tuple[str, ...]) -> dict[str, Any]:
    blob = f"{' '.join(tags)} {notes}".lower()
    transition = None
    match = re.search(r"(?:transition|recovery).*?\b([A-Z]{2,5})\b", notes)
    if match:
        transition = match.group(1)
    return {
        "requires_heat": "H" in requirements or "heat" in tags,
        "requires_rage": "R" in requirements or "rage" in tags,
        "counter_hit_launcher": "counter hit" in blob and "launch" in blob,
        "launcher": "launcher" in tags or "launch" in blob,
        "tornado": "tornado" in tags or "tornado" in blob,
        "homing": "homing" in tags or "homing" in blob,
        "power_crush": "power_crush" in tags or "power crush" in blob,
        "high_crush": "high_crush" in tags or "high crush" in blob,
        "low_crush": "low_crush" in tags or "low crush" in blob,
        "parry": "parry" in tags or "parry" in blob or "sabaki" in blob,
        "heat_engager": "heat engager" in blob,
        "heat_smash": "heat smash" in blob,
        "rage_art": "rage art" in blob,
        "chip": "chip" in tags or "chip damage" in blob,
        "wall_break": "wall break" in blob,
        "floor_break": "floor_break" in tags or "floor break" in blob,
        "balcony_break": "balcony_break" in tags or "balcony break" in blob,
        "stance_transition": transition,
        "requirements": list(requirements),
    }


def _action_features(local_id: int, levels: list[str], damage: list[float],
                     startup: dict[str, int] | None, recovery: dict[str, int] | None,
                     block: dict[str, int] | None, tags: list[str], mechanics: dict[str, Any]) -> list[float]:
    features = [0.0] * ACTION_FEATURE_SIZE
    level_slots = {"high": 0, "mid": 1, "low": 2, "throw": 3, "special_mid": 4, "special_low": 5}
    for level in levels:
        if level in level_slots:
            features[level_slots[level]] = 1.0
    features[6] = min(1.0, sum(damage) / 100.0)
    features[7] = -1.0 if startup is None else min(1.0, startup["min"] / 60.0)
    features[8] = -1.0 if recovery is None else min(1.0, recovery["max"] / 90.0)
    features[9] = -1.0 if block is None else max(-1.0, min(1.0, block["min"] / 30.0))
    flags = ["launcher", "tornado", "homing", "power_crush", "high_crush", "low_crush",
             "parry", "heat_engager", "heat_smash", "rage_art", "chip", "wall_break",
             "floor_break", "balcony_break", "requires_heat", "requires_rage"]
    for index, flag in enumerate(flags, start=10):
        features[index] = float(bool(mechanics.get(flag)))
    digest = sha256(str(local_id).encode()).digest()
    for index in range(26, ACTION_FEATURE_SIZE):
        features[index] = digest[index - 26] / 127.5 - 1.0
    return features


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value

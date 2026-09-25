"""Exports per-move engine bindings for the scalar full-combat engine.

Joins, per move:
  * the compiled catalog (parsed command, hit levels, damage, frame data,
    mechanics, stances, situational requirements);
  * the raw source row (frame-advantage suffixes: a = launch, d = knockdown;
    frame windows written in the notes: low/high crush, power crush, parry
    state, floating state, intangibility);
  * measured geometry from data/measurements/<slug>.yaml, if any;
  * Practice validation status from data/validation/<slug>.json;
  * resource and throw data written in the notes: Heat role and Heat Dash
    advantage, chip damage (also in Heat), self-damage, recoverable-health
    restores and removal, recoverable-only damage, throw-break input, side
    switches, spikes, and parry/reversal immunity. An amount the notes mention
    without a number is written as "?", which the binder reports as missing;
  * character state: the stance a move ends in ("r28 IZU"), crouched endings
    ("r25 FC"), stance transitions on hit or block, sidestep moves, Heat-timer
    costs, parry levels and parry outcomes, and, for characters with
    data/character_modules/<slug>/state_machine.yaml, resource gains and
    install bonuses (Jun: Kazama Essence and Divine Aura).

Optional stance branches ("Enter GEN +0 +11g r18 with F") become variant rows
("jun:67~GEN", variant_of "jun:67") with the branch's own recovery and
advantage. Stance and resource rules are written to
full_combat_stances.csv and full_combat_resources.csv next to the output, and
curated combo routes (data/character_modules/<slug>/routes.yaml, commands
resolved to stable ids) to full_combat_routes.csv.

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
    "heat_burst", "heat_smash", "heat_dash_block", "heat_dash_hit", "heat_dash_effect",
    "chip_block", "chip_block_heat", "recoverable_only", "removes_recoverable", "armor_damage_recoverable",
    "self_damage", "self_recoverable", "self_damage_without_heat",
    "restore_health_hit", "restore_recoverable_hit", "restore_recoverable_block",
    "throw_break", "side_switch_on_hit", "side_switch_on_break", "spike", "unparryable", "reversal_break",
    "variant_of", "result_crouching", "result_stance_on_hit", "result_stance_on_block",
    "requires_sidestep", "requires_running", "parry_levels", "parry_outcomes", "heat_cost_frames",
    "resource_gain_start", "resource_gain_hit", "resource_gain_airborne_hit", "resource_gain_block",
    "resource_gain_heat_activation", "install_damage_bonus", "install_chip", "install_range",
    "attack_throw", "attack_throw_front_only", "attack_throw_standing_only", "attack_throw_airborne",
    "attack_throw_damage", "back_turned_hit_advantage", "back_turned_hit_effect", "result_back_turned",
    "heat_parry", "heat_parry_levels", "heat_parry_outcome", "recoverable_damage", "cannot_ko",
    "rage_art_max_damage", "ki_charge",
    "practice_status",
] + [f"measured_{key}" for key in MEASUREMENT_KEYS]
UNKNOWN = "?"  # the notes state the property but not its amount
NOTE_MEASUREMENT_KEYS = ("chip_block", "chip_block_heat", "self_damage", "self_recoverable",
                         "restore_health_hit", "restore_recoverable_hit", "restore_recoverable_block", "throw_break")


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


def note_lines(notes: str) -> list[str]:
    return [line.strip("* ").strip() for line in notes.splitlines() if line.strip("* ").strip()]


def _numbers(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", text)]


def parse_heat_dash(lines: list[str]) -> dict[str, object]:
    """'Heat Dash +5, +43d (+35)' -> block 5, hit 43, knockdown. Hit-only forms leave block empty."""
    result: dict[str, object] = {"heat_dash_block": "", "heat_dash_hit": "", "heat_dash_effect": ""}
    for line in lines:
        if not re.match(r"heat dash\s*[+-]", line, re.IGNORECASE):
            continue
        body = re.sub(r"\([^)]*\)", "", line[len("heat dash"):])
        for part in body.split(","):
            match = re.search(r"([+-]\d+)([ad]?)", part)
            if not match:
                continue
            value, suffix = int(match.group(1)), match.group(2)
            lower = part.lower()
            on_block = "on block" in lower or (not suffix and "on hit" not in lower
                                               and result["heat_dash_block"] == "")
            if on_block:
                result["heat_dash_block"] = value
            else:
                result["heat_dash_hit"] = value
                result["heat_dash_effect"] = EFFECTS.get(suffix, "stun")
        break
    return result


def parse_chip(lines: list[str]) -> dict[str, str]:
    """Chip damage on block, per hit when the notes list one value per hit ("Chip damage (2,7)")."""
    block, heat = "", ""
    for line in lines:
        lower = line.lower()
        if "chip" not in lower or any(skip in lower for skip in ("absorb", "on hit", "heat dash", "scaling", "perfect")):
            continue
        if re.match(r"^[a-z ]+:", lower):  # conditional on a character resource ("Divine Aura: ...")
            continue
        text = re.sub(r"\([^)]*:[^)]*\)", "", line)  # "(DA:11)" alternatives
        values = _numbers(text)
        value = "|".join(str(number) for number in values) if values else UNKNOWN
        if "in heat" in lower:
            heat = value
        elif "on block" in lower or re.search(r"chip (?:damage )?(?:on block )?[(\[]", lower) \
                or re.fullmatch(r"\d+ chip damage", lower) or lower == "chip damage":
            block = block or value
    return {"chip_block": block, "chip_block_heat": heat}


def parse_recoverable(lines: list[str]) -> dict[str, object]:
    result: dict[str, object] = {
        "recoverable_only": 0, "removes_recoverable": 0, "armor_damage_recoverable": 0,
        "self_damage": "", "self_recoverable": "", "self_damage_without_heat": 0,
        "restore_health_hit": "", "restore_recoverable_hit": "", "restore_recoverable_block": "",
    }
    for line in lines:
        lower = line.lower()
        if re.search(r"only deals recoverable damage|deals recoverable damage only", lower):
            result["recoverable_only"] = 1
        if lower == "removes recoverable health":
            result["removes_recoverable"] = 1
        if "damage taken during power crush state is recoverable" in lower:
            result["armor_damage_recoverable"] = 1
        if "to self" in lower:
            values = _numbers(re.sub(r"^\d+(st|nd|rd|th) hit", "", lower))
            if "recoverable damage to self" in lower:
                amount = values[0] if values else UNKNOWN
                result["self_damage"], result["self_recoverable"] = amount, amount
            elif len(values) >= 2:
                result["self_damage"], result["self_recoverable"] = values[0], values[1]
            elif len(values) == 1:
                result["self_damage"], result["self_recoverable"] = values[0], 0
            else:
                result["self_damage"] = result["self_recoverable"] = UNKNOWN
            result["self_damage_without_heat"] = int("without heat" in lower)
        if lower.startswith("restores") and not any(skip in lower for skip in ("parry", "heat time", "perfect", "each")):
            health = re.search(r"(\d+) health", lower)
            recoverable = re.search(r"(\d+) (?:additional )?recoverable", lower)
            on_block = re.search(r"\((\d+) on block\)", lower)
            if health:
                result["restore_health_hit"] = int(health.group(1))
            if recoverable:
                result["restore_recoverable_hit"] = int(recoverable.group(1))
                if on_block:
                    result["restore_recoverable_block"] = int(on_block.group(1))
                elif "hit or block" in lower:
                    result["restore_recoverable_block"] = int(recoverable.group(1))
            elif "recoverable" in lower:
                result["restore_recoverable_hit"] = UNKNOWN
    return result


def parse_throw_break(lines: list[str]) -> str:
    """'1', '2', '1+2', '1|2' (either), 'none', '?' (depends on input), or '' (not stated)."""
    for line in lines:
        lower = line.lower()
        if lower.startswith("unbreakable") or "cannot throw break" in lower or "throw break: none" in lower:
            return "none"
        if "throw break" not in lower:
            continue
        if "depending" in lower:
            return UNKNOWN
        match = re.search(r"throw break:?\s*(\d(?:\+\d)?)(?:\s*or\s*(\d))?", lower) or \
            re.search(r"(\d(?:\+\d)?)(?:\s*or\s*(\d))?\s*throw break", lower)
        if match:
            return f"{match.group(1)}|{match.group(2)}" if match.group(2) else match.group(1)
    return ""


def parse_flags(lines: list[str]) -> dict[str, int]:
    flags = {"side_switch_on_hit": 0, "side_switch_on_break": 0, "spike": 0, "unparryable": 0, "reversal_break": 0}
    for line in lines:
        lower = line.lower().rstrip(".")
        switch = re.search(r"(?:^|\. )side switch(.*)", lower)
        if switch and not lower.startswith(("can ", "sometimes")):
            rest = switch.group(1)
            flags["side_switch_on_break"] |= int("break" in rest)
            flags["side_switch_on_hit"] |= int("hit" in rest or "break" not in rest)
        flags["spike"] |= int(lower == "spike")
        flags["unparryable"] |= int("unparryable" in lower)
        flags["reversal_break"] |= int(lower == "reversal break")
    return flags


STANCE_FIELDS = ["character", "stance", "can_guard", "max_frames", "auto_parry", "parry_outcome_high",
                 "parry_outcome_mid", "parry_outcome_low", "parry_outcome_throw", "pulse_interval_frames",
                 "pulse_recoverable", "pulse_resource", "status"]
ROUTE_FIELDS = ["character", "route", "category", "source", "distance", "near_wall", "opponent", "opponent_move",
                "heat", "resource", "step", "input", "delay", "expected_hits", "counter_hit_starter",
                "heat_activated"]
ROUTE_CATEGORIES = ("midscreen", "wall", "heat", "counter_hit")
RESOURCE_FIELDS = ["character", "name", "max", "install_threshold", "install_name", "persists_across_rounds",
                   "consumed_on_install", "status"]
LEVEL_CLASSES = ("high", "mid", "low", "throw")
NOT_STANCES = {"H", "R", "WS", "SS", "FC", "HFC", "WR", "BT"}


def load_state_machine(path: Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}


def result_state(recovery_text: object, stances: set[str]) -> tuple[str, int]:
    """'r28 IZU' -> ('IZU', 0); 'r25 FC' -> ('', 1)."""
    match = re.search(r"(?:^|\s)(?:r\d+(?:~\d+)?\s+)?([A-Za-z]{2,5})\s*$", str(recovery_text or "").strip())
    token = match.group(1).upper() if match else ""
    if token == "FC":
        return "", 1
    if token == "BT":
        return "BT", 0
    return (token if token in stances else ""), 0


def stance_transitions(lines: list[str], stances: set[str]) -> tuple[dict[str, str], list[str]]:
    """Automatic stance transitions on hit or block, and the lines they consumed."""
    result = {"result_stance_on_hit": "", "result_stance_on_block": ""}
    used = []
    for line in lines:
        match = re.match(r"transitions? to (?:r\d+ )?([A-Za-z]+) on (hit only|hit or block|hit|block)$", line, re.I)
        if not match or match.group(1).upper() not in stances:
            continue
        stance, when = match.group(1).upper(), match.group(2).lower()
        if "hit" in when:
            result["result_stance_on_hit"] = stance
        if "block" in when:
            result["result_stance_on_block"] = stance
        used.append(line)
    return result, used


def stance_variants(lines: list[str], stances: set[str]) -> list[dict[str, object]]:
    """Optional branches held with a direction: 'Enter GEN +0 +11g r18 with F',
    'Transition to +9, +26a (+16) GEN with F', 'Transition to r24 FC with D'."""
    variants = []
    for line in lines:
        lower = line.lower()
        # Input timing ("... with F on frame 21 with 10F delay") is not a condition.
        lower = re.sub(r"\s+on frame \d+(?:\s+with \d+f delay)?$", "", lower)
        if " on " in lower:  # conditional ("with D on whiff or block")
            continue
        enter = re.match(r"(?:enter|transition to) ([a-z]+)\s+([+-]\d+[a-z]*)?,?\s*([+-]\d+[a-z]*)?\s*r(\d+)\s+wi+th\s+(\S+)$", lower)
        advantage = re.match(r"transition to ([+-]\d+[a-z]*),\s*([+-]\d+[a-z]*)(?:\s*\([^)]*\))?\s+([a-z]+)\s+wi+th\s+(\S+)$", lower)
        crouch = re.match(r"transition(?: in)? to r?(\d+)\s+([a-z]+)\s+wi+th\s+(\S+)$", lower)
        plain = re.match(r"(?:enter|transition to) ([a-z]+)\s+wi+th\s+(\S+)$", lower)
        if enter:
            target, block, hit, recovery, held = enter.groups()
        elif advantage:
            block, hit, target, held = advantage.groups()
            recovery = None
        elif crouch:
            recovery, target, held = crouch.groups()
            block = hit = None
        elif plain:
            target, held = plain.groups()
            block = hit = recovery = None
        else:
            continue
        target = target.upper()
        if target not in {"FC", "BT"} and target not in stances:
            continue  # sidestep entries and unknown states are not branches the engine models
        variants.append({"target": target, "block": block, "hit": hit,
                         "recovery": int(recovery) if recovery else None, "input": held.upper()})
    return variants


def parse_resource(lines: list[str], name: str) -> dict[str, object]:
    result: dict[str, object] = {key: "" for key in (
        "resource_gain_start", "resource_gain_hit", "resource_gain_airborne_hit", "resource_gain_block",
        "resource_gain_heat_activation")}
    if not name:
        return result
    for line in lines:
        match = re.match(rf"gain (\d+) {re.escape(name.lower())}(?: on (.*))?$", line.lower())
        if not match:
            continue
        parts = re.split(r" and (\d+) on ", match.group(2) or "")
        segments = [(int(match.group(1)), parts[0])] + [(int(parts[index]), parts[index + 1])
                                                         for index in range(1, len(parts) - 1, 2)]
        for amount, text in segments:
            if not text:
                result["resource_gain_start"] = amount
            elif "heat" in text:
                result["resource_gain_heat_activation"] = amount
            else:
                generic_hit = "hit" in text and "normal" not in text and "airborne" not in text
                if "normal" in text or generic_hit:
                    result["resource_gain_hit"] = amount
                if "airborne" in text or generic_hit:
                    result["resource_gain_airborne_hit"] = amount
                if "block" in text:
                    result["resource_gain_block"] = amount
    return result


def parse_install(lines: list[str], prefixes: list[str]) -> dict[str, object]:
    result: dict[str, object] = {"install_damage_bonus": "", "install_chip": "", "install_range": ""}
    if not prefixes:
        return result
    alternatives = "|".join(re.escape(prefix.lower()) for prefix in prefixes)
    for line in lines:
        lower = line.lower()
        inline = re.search(rf"\((?:{alternatives})\s*:\s*(\d+)\)", lower)
        if inline and "chip" in lower:
            result["install_chip"] = int(inline.group(1))
        match = re.match(rf"(?:{alternatives})\s*:\s*(.*)$", lower)
        if not match:
            continue
        text = match.group(1)
        bonus = re.match(r"\+(\d+) damage", text)
        chip = re.match(r"(\d+) chip damage", text)
        reach = re.match(r"range increases to ([\d.]+)", text)
        if bonus:
            result["install_damage_bonus"] = int(bonus.group(1))
        elif chip:
            result["install_chip"] = int(chip.group(1))
        elif reach:
            result["install_range"] = float(reach.group(1))
    return result


def parse_parry_levels(lines: list[str]) -> str:
    """'Parries low punches or kicks' + 'Parries throws' -> 'low|throw'; '?' when not stated."""
    levels: list[str] = []
    for line in lines:
        lower = line.lower()
        # "Parries ...", "Parry mid and high ...", "Sabaki, parries ..."; not "Parry state", "Unparryable".
        if not re.search(r"(?:^|, )parr(?:y|ies)\b", lower) or lower.startswith("parry state"):
            continue
        found = [level for level in LEVEL_CLASSES if re.search(rf"\b{level}s?\b", lower)]
        if not found and re.search(r"\ball\b", lower):
            found = ["high", "mid", "low"]
        levels += [level for level in found if level not in levels]
    return "|".join(level for level in LEVEL_CLASSES if level in levels) or UNKNOWN


def parse_attack_throw(lines: list[str]) -> tuple[dict[str, object], list[str]]:
    """'Transition to attack throw on front standing or airborne hit' and the like."""
    result: dict[str, object] = {"attack_throw": "", "attack_throw_front_only": 0, "attack_throw_standing_only": 0,
                                 "attack_throw_airborne": 0, "attack_throw_damage": ""}
    for line in lines:
        match = re.match(r"transitions? to attack throw (?:automatically )?on ([a-z ]+?)(?: only)?(?:,\s*(.*))?$",
                         line.lower())
        if not match:
            continue
        condition, extra = match.group(1).strip(), match.group(2) or ""
        words = set(condition.split())
        if not words <= {"hit", "ch", "counter", "front", "standing", "or", "airborne"}:
            continue  # "on 1st hit", "after 2nd hit", timed inputs: not modeled
        result["attack_throw"] = "counter_hit" if words & {"ch", "counter"} else "hit"
        result["attack_throw_front_only"] = int("front" in words)
        result["attack_throw_standing_only"] = int("standing" in words)
        result["attack_throw_airborne"] = int("airborne" in words or "standing" not in words)
        bonus = re.search(r"\+(\d+) damage", extra)
        if bonus:
            result["attack_throw_damage"] = int(bonus.group(1))
        return result, [line]
    return result, []


def parse_back_turned_hit(lines: list[str]) -> dict[str, object]:
    """'Hit vs BT +10a (+1)' or '+10a (+1) and Balcony Break on BT hit'."""
    for line in lines:
        lower = line.lower()
        match = re.match(r"hit vs bt ([+-]\d+)([a-z]?)", lower) or \
            re.match(r"([+-]\d+)([a-z]?)(?:\s*\([^)]*\))?(?: and [a-z ]+)? on bt hit$", lower)
        if match:
            return {"back_turned_hit_advantage": int(match.group(1)),
                    "back_turned_hit_effect": EFFECTS.get(match.group(2), "stun")}
    return {"back_turned_hit_advantage": "", "back_turned_hit_effect": ""}


def parse_damage_rules(lines: list[str], name: str, command: str) -> dict[str, object]:
    result: dict[str, object] = {"heat_parry": "", "recoverable_damage": "", "cannot_ko": 0,
                                 "rage_art_max_damage": "", "ki_charge": int(name == "Ki Charge" or command == "1+2+3+4")}
    for line in lines:
        lower = line.lower()
        heat_parry = re.search(r"power up in heat \(ps(\d+)~(\d*)\)", lower)
        if heat_parry:
            result["heat_parry"] = f"{int(heat_parry.group(1))}~{int(heat_parry.group(2) or OPEN_WINDOW_END)}"
        partial = re.match(r"deals (\d+) recoverable damage(?: on hit)?(?:\s*-.*)?$", lower)
        if partial:
            result["recoverable_damage"] = int(partial.group(1))
        if re.search(r"cannot (?:cause a )?k\.?o", lower):
            result["cannot_ko"] = 1
        maximum = re.search(r"damage increases with lower health, maximum (\d+)", lower)
        if maximum:
            result["rage_art_max_damage"] = int(maximum.group(1))
    return result


def damage_parts(raw_damage: object, parsed: list) -> list:
    """'[12;12]' lists alternatives, not parts: keep the first of each alternative group."""
    text = str(raw_damage or "")
    if ";" not in text:
        return list(parsed)
    groups = re.sub(r"\[([^\]]*)\]", lambda match: match.group(1).split(";")[0], text)
    return [float(value) for value in re.findall(r"\d+(?:\.\d+)?", groups)]


def outcome_kind(command: str) -> str:
    match = re.search(r"\((high|mid|low|throw)\)\s*$", command, re.I)
    return match.group(1).lower() if match else ""


def load_measurements(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Measurements marked stale by tools/patch_update.py (the move changed) count as missing.
    return {str(key): dict(value or {}) for key, value in (document.get("moves") or {}).items()
            if not (value or {}).get("stale")}


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
                "__machine__": load_state_machine(data_root / "character_modules" / slug / "state_machine.yaml"),
                "__routes__": data_root / "character_modules" / slug / "routes.yaml",
                "__stances__": {str(value).upper() for value in source.get("stances") or []} - NOT_STANCES,
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
        lines = note_lines(str(raw.get("notes") or ""))
        machine = cache["__machine__"]
        resource = machine.get("resource") or {}
        stances = cache["__stances__"] | {str(name).upper() for name in (machine.get("stances") or {})}
        result_stance, result_crouching = result_state(raw.get("recovery"), stances)
        result_back_turned = int(result_stance == "BT" or any(line.lower() == "transition to bt" for line in lines))
        result_stance = "" if result_stance == "BT" else result_stance
        transitions, consumed = stance_transitions(lines, stances)
        attack_throw, throw_lines = parse_attack_throw(lines)
        consumed = consumed + throw_lines + [line for line in lines if line.lower() == "transition to bt"]
        remaining_notes = "\n".join(line for line in lines if line not in consumed)
        legal_state = move["legal_state"]
        row = {
            "stable_id": move["stable_id"], "character": slug, "name": one_line(move["name"]),
            "command": one_line(move["command"]["raw"]), "parser_status": move["command"]["parser_status"],
            "source_consistency": move["validation"]["source_consistency"],
            "reactive": int(any(str(step.get("requirement") or "").startswith("PARRY_SUCCESS") for step in steps)),
            "hit_levels": join(move["hit_levels"]), "damages": join(damage_parts(raw.get("damage"), move["damage"])),
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
            "result_stance": result_stance,
            "automatic_transition": automatic_transitions(remaining_notes),
            "practice_status": (cache["__ledger__"].get(move["stable_id"]) or {}).get("status", "pending"),
            **{name: "" for name in WINDOW_PATTERNS},
            **parse_windows(str(raw.get("notes") or "")),
            "heat_burst": int(mechanics.get("heat_burst", False)), "heat_smash": int(mechanics["heat_smash"]),
            **parse_heat_dash(lines), **parse_chip(lines), **parse_recoverable(lines),
            "throw_break": parse_throw_break(lines) if "throw" in move["hit_levels"] else "",
            **parse_flags(lines),
            "variant_of": "", "result_crouching": result_crouching, **transitions,
            "requires_sidestep": int(bool(legal_state.get("requires_sidestep"))),
            "requires_running": int(bool(legal_state.get("requires_running"))),
            "parry_levels": parse_parry_levels(lines) if parse_windows(str(raw.get("notes") or "")).get("parry") else "",
            "parry_outcomes": "",
            "heat_cost_frames": next((int(match.group(1)) for line in lines
                                      if (match := re.search(r"consumes (\d+)f of (?:remaining )?heat tim(?:e|er)", line, re.I))), ""),
            **parse_resource(lines, str(resource.get("name") or "")),
            **parse_install(lines, list(resource.get("install_note_prefixes") or [])),
            **attack_throw, **parse_back_turned_hit(lines), "result_back_turned": result_back_turned,
            "heat_parry_levels": "", "heat_parry_outcome": "",
            **parse_damage_rules(lines, one_line(move["name"]), one_line(move["command"]["raw"])),
        }
        for key in MEASUREMENT_KEYS:
            value = measured.get(key)
            row[f"measured_{key}"] = join(value) if isinstance(value, list) else ("" if value is None else value)
        # Measured amounts replace what the notes leave unstated ("?").
        for key in NOTE_MEASUREMENT_KEYS:
            if measured.get(key) is not None:
                value = measured[key]
                row[key] = join(value) if isinstance(value, list) else value
        rows.append(row)
        for variant in stance_variants(lines, stances):
            branch = dict(row)
            branch.update({
                "stable_id": f"{move['stable_id']}~{variant['target']}", "variant_of": move["stable_id"],
                "command": f"{row['command']}~{variant['input']}", "result_stance_on_hit": "",
                "result_stance_on_block": "", "result_stance": "", "result_crouching": 0, "result_back_turned": 0,
            })
            if variant["target"] == "FC":
                branch["result_crouching"] = 1
            elif variant["target"] == "BT":
                branch["result_back_turned"] = 1
            else:
                branch["result_stance"] = variant["target"]
            if variant["recovery"] is not None:
                branch["recovery"] = variant["recovery"]
            if variant["block"] is not None or variant["hit"] is not None:
                block_value, _, _ = parse_advantage(variant["block"])
                hit_value, hit_kind, _ = parse_advantage(variant["hit"])
                branch["block_advantage"] = "" if block_value is None else block_value
                branch["hit_advantage"] = "" if hit_value is None else hit_value
                branch["hit_effect"] = hit_kind
            elif variant["recovery"] is not None and row["recovery"] != "":
                # Only the attacker's recovery changes, so both advantages shift by the difference.
                shift = int(row["recovery"]) - variant["recovery"]
                for key in ("block_advantage", "hit_advantage", "counter_hit_advantage"):
                    if row[key] != "":
                        branch[key] = int(row[key]) + shift
            rows.append(branch)
    link_parry_outcomes(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_character_rules(rows, source_cache, output.parent)
    return len(rows)


def link_parry_outcomes(rows: list[dict]) -> None:
    """'b+1+3,P' is the outcome of b+1+3's parry, for every level it parries;
    '(Low)'/'(Throw)' outcomes apply to that level only."""
    by_command = {(row["character"], row["command"]): row for row in rows if not row["variant_of"]}
    for row in rows:
        if not int(row["reactive"]) or row["variant_of"]:
            continue
        base_command = re.sub(r",\s*P(\s*\([^)]*\))?\s*$", "", row["command"])
        heat_base = by_command.get((row["character"], base_command[2:])) if base_command.startswith("H.") else None
        if heat_base is not None and heat_base["heat_parry"]:
            heat_base["heat_parry_outcome"] = row["stable_id"]
            heat_base["heat_parry_levels"] = row["parry_levels"] or UNKNOWN
            continue
        base = by_command.get((row["character"], base_command))
        if base is None or base is row:
            continue
        slots = (base["parry_outcomes"] or "|||").split("|")
        kind = outcome_kind(row["command"])
        for index, level in enumerate(LEVEL_CLASSES):
            if not kind or kind == level:
                slots[index] = row["stable_id"]
        base["parry_outcomes"] = "|".join(slots)


def write_character_rules(rows: list[dict], source_cache: dict, directory: Path) -> None:
    ids = {(row["character"], row["command"]): row["stable_id"] for row in rows if not row["variant_of"]}
    stance_rows, resource_rows = [], []
    for slug, cache in sorted(source_cache.items()):
        machine = cache["__machine__"]
        for name, rule in (machine.get("stances") or {}).items():
            rule = rule or {}
            outcomes = rule.get("parry_outcomes") or {}
            stance_rows.append({
                "character": slug, "stance": str(name).upper(), "can_guard": int(bool(rule.get("can_guard", True))),
                "max_frames": "" if rule.get("max_frames") is None else rule["max_frames"],
                "auto_parry": join(level for level in LEVEL_CLASSES if level in (rule.get("auto_parry") or [])),
                **{f"parry_outcome_{level}": ids.get((slug, str(outcomes[level])), "") if level in outcomes else ""
                   for level in LEVEL_CLASSES},
                "pulse_interval_frames": rule.get("pulse_interval_frames") or 0,
                "pulse_recoverable": rule.get("pulse_recoverable") or 0,
                "pulse_resource": rule.get("pulse_resource") or 0,
                "status": one_line(json.dumps(rule.get("status") or {}, sort_keys=True)),
            })
            missing = [str(command) for command in outcomes.values() if (slug, str(command)) not in ids]
            if missing:
                raise ValueError(f"{slug} stance {name}: parry outcome not in the catalog: {missing}")
        resource = machine.get("resource")
        if resource:
            resource_rows.append({
                "character": slug, "name": resource.get("name", ""), "max": resource.get("max", 0),
                "install_threshold": "" if resource.get("install_threshold") is None else resource["install_threshold"],
                "install_name": resource.get("install_name", ""),
                "persists_across_rounds": int(bool(resource.get("persists_across_rounds"))),
                "consumed_on_install": int(bool(resource.get("consumed_on_install"))),
                "status": resource.get("status", ""),
            })
    route_rows = []
    by_command = {(row["character"], row["command"]): row["stable_id"] for row in rows}
    for slug in sorted(source_cache):
        path = source_cache[slug]["__routes__"]
        if not path.exists():
            continue
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for route in document.get("routes") or []:
            name, category = str(route["name"]), str(route["category"])
            if category not in ROUTE_CATEGORIES:
                raise ValueError(f"{slug} route {name}: unknown category {category}")
            setup, expect = route.get("setup") or {}, route.get("expect") or {}

            def resolve(command: str) -> str:
                if command.startswith("@"):
                    return command  # universal action, e.g. @HeatDash
                stable_id = by_command.get((slug, command))
                if stable_id is None:
                    raise ValueError(f"{slug} route {name}: command not in the catalog: {command}")
                return stable_id

            opponent_move = setup.get("opponent_move")
            for index, step in enumerate(route.get("steps") or []):
                command, delay = (step, 0) if not isinstance(step, dict) else (step["input"], step.get("delay", 0))
                route_rows.append({
                    "character": slug, "route": name, "category": category, "source": route.get("source", ""),
                    "distance": setup.get("distance", 0.8), "near_wall": int(bool(setup.get("near_wall"))),
                    "opponent": setup.get("opponent", "walk"),
                    "opponent_move": resolve(str(opponent_move)) if opponent_move else "",
                    "heat": int(bool(setup.get("heat"))), "resource": setup.get("resource", 0),
                    "step": index, "input": resolve(str(command)), "delay": int(delay),
                    "expected_hits": expect.get("hits", ""),
                    "counter_hit_starter": int(bool(expect.get("counter_hit_starter"))),
                    "heat_activated": int(bool(expect.get("heat_activated"))),
                })
    for name, fields, values in (("full_combat_stances.csv", STANCE_FIELDS, stance_rows),
                                 ("full_combat_resources.csv", RESOURCE_FIELDS, resource_rows),
                                 ("full_combat_routes.csv", ROUTE_FIELDS, route_rows)):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)


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

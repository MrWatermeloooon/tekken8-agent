from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import random
from typing import Any, Iterable

import yaml


PROBABILITY_FIELDS = (
    "aggression", "input_error_rate", "approach", "backdash",
    "sidestep_left", "sidestep_right", "low_frequency", "throw_frequency",
    "delay_frequency", "stance_entry_frequency", "heat_usage",
    "punish_accuracy", "throw_break_accuracy", "low_block_accuracy",
)

GROUP_BITS = {
    "fundamentals": 1 << 0,
    "rushdown": 1 << 1,
    "stance_heavy": 1 << 2,
    "grappler": 1 << 3,
    "keep_out": 1 << 4,
    "evasive": 1 << 5,
    "specialist": 1 << 6,
    "counter_hit": 1 << 7,
    "movement": 1 << 8,
}


@dataclass(frozen=True)
class Character:
    id: int
    slug: str
    name: str
    groups: tuple[str, ...]
    specialty: str
    sidestep: str
    signatures: tuple[str, ...]


@dataclass(frozen=True)
class Archetype:
    id: int
    key: str
    base: dict[str, Any]


@dataclass(frozen=True)
class OpponentProfile:
    id: int
    name: str
    character_id: int
    character_slug: str
    group_mask: int
    archetype_id: int
    archetype: str
    variation_id: int
    variation: str
    aggression: float
    reaction_min: int
    reaction_max: int
    input_error_rate: float
    approach: float
    backdash: float
    sidestep_left: float
    sidestep_right: float
    low_frequency: float
    throw_frequency: float
    delay_frequency: float
    stance_entry_frequency: float
    heat_usage: float
    punish_accuracy: float
    throw_break_accuracy: float
    low_block_accuracy: float

    def episode_variant(self, seed: int, episode: int) -> "OpponentProfile":
        """Deterministically jitter behavior so a profile is a distribution."""
        rng = random.Random((seed << 32) ^ episode ^ self.id)
        changes: dict[str, Any] = {}
        for field in PROBABILITY_FIELDS:
            radius = 0.012 if field == "input_error_rate" else 0.035
            changes[field] = _clamp(getattr(self, field) + rng.uniform(-radius, radius))
        shift = rng.randint(-2, 2)
        changes["reaction_min"] = max(1, self.reaction_min + shift)
        changes["reaction_max"] = max(changes["reaction_min"], self.reaction_max + shift)
        return replace(self, **changes)


@dataclass(frozen=True)
class OpponentCatalog:
    as_of: str
    source: str
    characters: tuple[Character, ...]
    archetypes: tuple[Archetype, ...]
    variations: tuple[str, ...]
    profiles: tuple[OpponentProfile, ...]
    announced_unreleased: tuple[str, ...]

    def character(self, slug_or_id: str | int) -> Character:
        for character in self.characters:
            if character.id == slug_or_id or character.slug == slug_or_id:
                return character
        raise KeyError(f"unknown character: {slug_or_id}")

    def profiles_for(self, character: str | int, archetype: str | None = None) -> tuple[OpponentProfile, ...]:
        selected = self.character(character)
        return tuple(
            profile for profile in self.profiles
            if profile.character_id == selected.id
            and (archetype is None or profile.archetype == archetype)
        )


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def load_catalog(data_root: str | Path | None = None) -> OpponentCatalog:
    root = Path(data_root) if data_root is not None else default_data_root()
    roster = _read_yaml(root / "roster.yaml")
    styles = _read_yaml(root / "opponent_archetypes.yaml")
    characters = tuple(
        Character(
            id=int(row["id"]), slug=str(row["slug"]), name=str(row["name"]),
            groups=tuple(str(value) for value in row["groups"]),
            specialty=str(row["specialty"]), sidestep=str(row["sidestep"]),
            signatures=tuple(str(value) for value in row["signatures"]),
        ) for row in roster["characters"]
    )
    archetypes = tuple(
        Archetype(id=int(row["id"]), key=str(row["key"]), base=dict(row["base"]))
        for row in styles["archetypes"]
    )
    variations = tuple(str(value) for value in styles["variation_names"])
    _validate_catalog_inputs(roster, styles, characters, archetypes, variations)
    profiles = tuple(_generate_profiles(characters, archetypes, variations, styles["variation_deltas"]))
    expected = len(characters) * int(styles["profiles_per_character"])
    if len(profiles) != expected:
        raise ValueError(f"generated {len(profiles)} profiles; expected {expected}")
    return OpponentCatalog(
        as_of=str(roster["as_of"]), source=str(roster["source"]),
        characters=characters, archetypes=archetypes, variations=variations,
        profiles=profiles,
        announced_unreleased=tuple(str(value) for value in roster.get("announced_unreleased", [])),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _validate_catalog_inputs(
    roster: dict[str, Any], styles: dict[str, Any],
    characters: tuple[Character, ...], archetypes: tuple[Archetype, ...],
    variations: tuple[str, ...],
) -> None:
    declared = int(roster["playable_count"])
    if len(characters) != declared:
        raise ValueError(f"roster declares {declared} characters but contains {len(characters)}")
    if [character.id for character in characters] != list(range(len(characters))):
        raise ValueError("character IDs must be contiguous and ordered from zero")
    if len({character.slug for character in characters}) != len(characters):
        raise ValueError("character slugs must be unique")
    if any(character.sidestep not in {"left", "right"} for character in characters):
        raise ValueError("every character needs a left/right sidestep tendency")
    if any(not character.groups or not character.signatures for character in characters):
        raise ValueError("every character needs groups and signature mechanics")
    if len(archetypes) != 10 or [value.id for value in archetypes] != list(range(10)):
        raise ValueError("the style matrix must contain exactly ten ordered archetypes")
    if len(variations) != 5 or len(set(variations)) != 5:
        raise ValueError("the style matrix must contain exactly five unique variations")
    if int(styles["profiles_per_character"]) != len(archetypes) * len(variations):
        raise ValueError("profiles_per_character must equal archetypes times variations")
    for archetype in archetypes:
        missing = set(PROBABILITY_FIELDS) - set(archetype.base)
        if missing:
            raise ValueError(f"archetype {archetype.key} is missing {sorted(missing)}")
        reaction = archetype.base.get("reaction_frames")
        if not isinstance(reaction, list) or len(reaction) != 2:
            raise ValueError(f"archetype {archetype.key} needs reaction_frames [min, max]")


def _generate_profiles(
    characters: Iterable[Character], archetypes: Iterable[Archetype],
    variations: tuple[str, ...], variation_deltas: dict[str, dict[str, float]],
) -> Iterable[OpponentProfile]:
    for character in characters:
        for archetype in archetypes:
            for variation_id, variation in enumerate(variations):
                yield _build_profile(character, archetype, variation_id, variation, variation_deltas[variation])


def _build_profile(
    character: Character, archetype: Archetype, variation_id: int,
    variation: str, delta: dict[str, float],
) -> OpponentProfile:
    values = dict(archetype.base)
    group_bias = _group_biases(character.groups)
    stable = sha256(f"{character.slug}:{archetype.key}:{variation}".encode()).digest()
    for index, field in enumerate(PROBABILITY_FIELDS):
        jitter = ((stable[index] / 255.0) - 0.5) * 0.04
        value = float(values[field]) + group_bias.get(field, 0.0) + jitter
        if field == "aggression": value += float(delta["aggression"])
        elif field == "input_error_rate": value += float(delta["error"])
        elif field == "delay_frequency": value += float(delta["delay"])
        elif field in {"punish_accuracy", "throw_break_accuracy", "low_block_accuracy"}:
            value += float(delta["defense"])
        values[field] = _clamp(value)
    reaction = values["reaction_frames"]
    reaction_min = max(1, int(reaction[0]) + int(delta["reaction"]))
    reaction_max = max(reaction_min, int(reaction[1]) + int(delta["reaction"]))
    profile_id = character.id * 50 + archetype.id * 5 + variation_id
    return OpponentProfile(
        id=profile_id, name=f"{character.slug}_{archetype.key}_{variation_id + 1:02d}",
        character_id=character.id, character_slug=character.slug,
        group_mask=sum(GROUP_BITS.get(group, 0) for group in character.groups),
        archetype_id=archetype.id, archetype=archetype.key,
        variation_id=variation_id, variation=variation,
        reaction_min=reaction_min, reaction_max=reaction_max,
        **{field: float(values[field]) for field in PROBABILITY_FIELDS},
    )


def _group_biases(groups: tuple[str, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    table: dict[str, dict[str, float]] = {
        "rushdown": {"aggression": 0.07, "approach": 0.06, "backdash": -0.04},
        "fundamentals": {"punish_accuracy": 0.05, "low_block_accuracy": 0.03},
        "grappler": {"throw_frequency": 0.18, "approach": 0.04},
        "keep_out": {"backdash": 0.08, "approach": -0.05},
        "stance_heavy": {"stance_entry_frequency": 0.16, "delay_frequency": 0.04},
        "evasive": {"sidestep_left": 0.05, "sidestep_right": 0.05},
        "counter_hit": {"punish_accuracy": 0.05, "delay_frequency": 0.05},
        "movement": {"backdash": 0.04, "sidestep_left": 0.04, "sidestep_right": 0.04},
        "specialist": {"stance_entry_frequency": 0.06, "heat_usage": 0.07},
    }
    for group in groups:
        for field, adjustment in table.get(group, {}).items():
            result[field] = result.get(field, 0.0) + adjustment
    return result


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))

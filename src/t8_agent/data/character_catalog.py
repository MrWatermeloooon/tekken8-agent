from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CatalogMove:
    move_id: str
    command: str
    hit_level: str
    damage: str
    startup: str
    block: str
    hit: str
    counter_hit: str = ""
    tags: tuple[str, ...] = ()
    tier: str = "specialist"
    notes: str = ""


@dataclass(frozen=True)
class ComboSpec:
    combo_id: str
    starter: str
    route: tuple[str, ...]
    difficulty: str
    purpose: str
    notes: str = ""


@dataclass(frozen=True)
class CharacterCatalog:
    character_id: str
    display_name: str
    game: str
    source_url: str
    source_note: str
    moves: tuple[CatalogMove, ...]
    combos: tuple[ComboSpec, ...]

    @property
    def core_moves(self) -> tuple[CatalogMove, ...]:
        return tuple(move for move in self.moves if move.tier == "core")


def load_character_catalog(path: str | Path) -> CharacterCatalog:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    moves = tuple(_load_move(item) for item in data.get("moves", []))
    combos = tuple(_load_combo(item) for item in data.get("combos", []))
    catalog = CharacterCatalog(
        character_id=str(data["character_id"]),
        display_name=str(data["display_name"]),
        game=str(data.get("game", "tekken_8")),
        source_url=str(data.get("source_url", "")),
        source_note=str(data.get("source_note", "")),
        moves=moves,
        combos=combos,
    )
    _validate_catalog(catalog)
    return catalog


def _load_move(item: dict[str, Any]) -> CatalogMove:
    return CatalogMove(
        move_id=str(item["id"]),
        command=str(item["command"]),
        hit_level=str(item.get("hit_level", "")),
        damage=str(item.get("damage", "")),
        startup=str(item.get("startup", "")),
        block=str(item.get("block", "")),
        hit=str(item.get("hit", "")),
        counter_hit=str(item.get("counter_hit", "")),
        tags=tuple(str(tag) for tag in item.get("tags", [])),
        tier=str(item.get("tier", "specialist")),
        notes=str(item.get("notes", "")),
    )


def _load_combo(item: dict[str, Any]) -> ComboSpec:
    return ComboSpec(
        combo_id=str(item["id"]),
        starter=str(item["starter"]),
        route=tuple(str(step) for step in item.get("route", [])),
        difficulty=str(item.get("difficulty", "unknown")),
        purpose=str(item.get("purpose", "")),
        notes=str(item.get("notes", "")),
    )


def _validate_catalog(catalog: CharacterCatalog) -> None:
    move_ids = [move.move_id for move in catalog.moves]
    duplicate_moves = sorted({move_id for move_id in move_ids if move_ids.count(move_id) > 1})
    if duplicate_moves:
        raise ValueError(f"duplicate move ids in {catalog.character_id}: {duplicate_moves}")
    known = set(move_ids)
    for combo in catalog.combos:
        missing = [step for step in (combo.starter, *combo.route) if step not in known]
        if missing:
            raise ValueError(f"combo {combo.combo_id} references unknown move ids: {missing}")

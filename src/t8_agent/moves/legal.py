from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catalog import CompiledMoveCatalog


UNIVERSAL_ACTIONS = (
    "neutral", "walk_forward", "walk_back", "dash_forward", "dash_back", "crouch", "stand",
    "jump", "sidestep_left", "sidestep_right", "sidewalk_left", "sidewalk_right", "block_high",
    "block_low", "low_parry", "throw_break_1", "throw_break_2", "throw_break_1p2",
)


@dataclass(frozen=True)
class MoveRuntimeState:
    posture: str = "standing"
    stance: str | None = None
    heat: bool = False
    rage: bool = False
    busy: bool = False
    grounded: bool = False
    airborne: bool = False
    knocked_down: bool = False
    combo_active: bool = False
    resources: dict[str, float] = field(default_factory=dict)


def legal_action_mask(
    catalog: CompiledMoveCatalog,
    character: str | int,
    state: MoveRuntimeState,
    *,
    require_validated: bool = True,
) -> np.ndarray:
    character_row = _character(catalog, character)
    moves = [move for move in catalog.moves if move["character_id"] == character_row["id"]]
    mask = np.zeros(len(UNIVERSAL_ACTIONS) + len(moves), dtype=bool)
    mask[0] = True
    if not state.busy:
        mask[1:len(UNIVERSAL_ACTIONS)] = True
    if state.airborne or state.knocked_down:
        mask[1:len(UNIVERSAL_ACTIONS)] = False
    if state.grounded or state.knocked_down:
        mask[6] = not state.busy
    if state.busy:
        return mask
    for local_id, move in enumerate(moves):
        legal = move["command"]["parser_status"] == "parsed"
        legal = legal and move["validation"]["source_consistency"] == "valid"
        if require_validated:
            legal = legal and move["validation"]["simulation"] == "ready"
            legal = legal and move["validation"]["practice"] == "pass"
        rule = move["legal_state"]
        if rule["posture"] not in {"standing", state.posture}:
            legal = False
        if rule["requires_heat"] and not state.heat:
            legal = False
        if rule["requires_rage"] and not state.rage:
            legal = False
        if rule["stances"] and (state.stance or "").upper() not in rule["stances"]:
            legal = False
        mask[len(UNIVERSAL_ACTIONS) + local_id] = legal
    return mask


def _character(catalog: CompiledMoveCatalog, value: str | int) -> dict:
    for row in catalog.characters:
        if row["slug"] == value or row["id"] == value:
            return row
    raise ValueError(f"unknown released character: {value}")

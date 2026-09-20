from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .catalog import CompiledMoveCatalog


REQUIRED_ROUTE_CATEGORIES = ("midscreen", "wall", "heat", "counter_hit")
REQUIRED_SCENARIOS = (
    "hit", "block", "counter_hit", "whiff", "transition", "resource",
    "cpu_catalog", "cuda_parity",
)


@dataclass(frozen=True)
class CharacterGateReport:
    character: str
    move_count: int
    ready: bool
    blockers: tuple[str, ...]
    parser_pending: int
    source_conflicts: int
    measurement_pending: int
    practice_pending: int
    scenario_pending: tuple[str, ...]
    route_pending: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_validation_ledger(path: str | Path) -> dict[str, Any]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {"schema_version": 1, "moves": {}, "scenarios": {}, "routes": {}}
    document = json.loads(ledger_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported validation ledger schema: {ledger_path}")
    return document


def evaluate_character_gate(
    catalog: CompiledMoveCatalog,
    character: str,
    validation_root: str | Path,
) -> CharacterGateReport:
    character_row = next((row for row in catalog.characters if row["slug"] == character), None)
    if character_row is None:
        raise ValueError(f"unknown character: {character}")
    moves = [move for move in catalog.moves if move["character_id"] == character_row["id"]]
    ledger = load_validation_ledger(Path(validation_root) / f"{character}.json")
    practice = ledger.get("moves") or {}
    scenarios = ledger.get("scenarios") or {}
    routes = ledger.get("routes") or {}
    parser_pending = sum(move["command"]["parser_status"] != "parsed" for move in moves)
    source_conflicts = sum(move["validation"]["source_consistency"] != "valid" for move in moves)
    measurement_pending = sum(
        any(value is None for value in move["measurements"].values()) for move in moves
    )
    practice_pending = sum(
        (practice.get(move["stable_id"]) or {}).get("status") != "pass" for move in moves
    )
    scenario_pending = tuple(name for name in REQUIRED_SCENARIOS if scenarios.get(name) != "pass")
    route_pending = tuple(name for name in REQUIRED_ROUTE_CATEGORIES if routes.get(name) != "pass")
    blockers: list[str] = []
    if not moves:
        blockers.append("missing documented move source")
    if parser_pending:
        blockers.append(f"{parser_pending} commands need parser review")
    if source_conflicts:
        blockers.append(f"{source_conflicts} source frame records are inconsistent")
    if measurement_pending:
        blockers.append(f"{measurement_pending} moves lack measured combat fields")
    if practice_pending:
        blockers.append(f"{practice_pending} moves need offline Practice validation")
    if scenario_pending:
        blockers.append("missing scenario gates: " + ", ".join(scenario_pending))
    if route_pending:
        blockers.append("missing route gates: " + ", ".join(route_pending))
    return CharacterGateReport(
        character=character,
        move_count=len(moves),
        ready=not blockers,
        blockers=tuple(blockers),
        parser_pending=parser_pending,
        source_conflicts=source_conflicts,
        measurement_pending=measurement_pending,
        practice_pending=practice_pending,
        scenario_pending=scenario_pending,
        route_pending=route_pending,
    )

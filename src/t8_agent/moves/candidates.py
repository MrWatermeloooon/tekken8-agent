"""Variable per-character action candidates for the full-move policy contract.

A fighter's candidates are the universal actions followed by every
non-reactive engine row of that fighter (catalog moves and their stance-branch
variants such as ``jun:67~GEN``), in the order of
``data/generated/full_move_candidates.csv``. tools/export_full_combat_bindings.py
writes that file, and both the CUDA trainer and the live runtime read it, so
candidate indices mean the same thing everywhere.

Each candidate has a fixed ACTION_FEATURE_SIZE property vector; the parametric
policy scores a state query against these vectors (the CUDA
``parametric_policy_output_kernel`` computes ``dot(query, features) /
sqrt(feature_size)``), so the same weights serve every fighter and every
candidate count. Rows the engine cannot bind, and universal actions it has no
mechanic for, stay in the list and are always masked: indices never shift.

Universal actions have no frame data. Their vectors leave the property slots
at zero, mark startup/recovery/block unknown (-1, as for undocumented moves),
and carry an identity hash in the last six slots like moves do.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import ACTION_FEATURE_SIZE, CompiledMoveCatalog
from .legal import UNIVERSAL_ACTIONS
from .notation import parse_command


FULL_MOVE_ACTION_CONTRACT = "full-move-v1"
UNIVERSAL_PREFIX = "universal:"
# The live/compatibility universal actions plus the engine actions a full-move
# policy needs: Heat Dash after an engager, getting up, and teching a landing.
FULL_MOVE_UNIVERSAL_ACTIONS = UNIVERSAL_ACTIONS + ("heat_dash", "wakeup_stand", "wakeup_roll", "tech_roll")
# Upper bound used for memory planning: the largest candidate list (Yoshimitsu, 354)
# rounded up to a multiple of 16.
MAX_CANDIDATES = 368
CANDIDATES_FILE = "full_move_candidates.csv"
CANDIDATE_FIELDS = ["character", "candidate", "stable_id", "kind", "command"] + [
    f"feature_{index}" for index in range(ACTION_FEATURE_SIZE)]


@dataclass(frozen=True)
class CandidateTable:
    character: str
    stable_ids: tuple[str, ...]
    kinds: tuple[str, ...]                        # universal, move, or variant
    commands: tuple[dict[str, Any] | None, ...]   # None for universal actions
    features: np.ndarray                          # float32 [candidates, ACTION_FEATURE_SIZE]

    @property
    def universal_count(self) -> int:
        return len(FULL_MOVE_UNIVERSAL_ACTIONS)

    def __len__(self) -> int:
        return len(self.stable_ids)

    def is_universal(self, index: int) -> bool:
        return index < self.universal_count

    def index_of(self, stable_id: str) -> int:
        try:
            return self.stable_ids.index(stable_id)
        except ValueError as exc:
            raise KeyError(f"{stable_id} is not a candidate of {self.character}") from exc


def identity_features(key: str) -> list[float]:
    digest = sha256(key.encode()).digest()
    return [digest[index] / 127.5 - 1.0 for index in range(ACTION_FEATURE_SIZE - 26)]


def universal_action_features(name: str) -> list[float]:
    if name not in FULL_MOVE_UNIVERSAL_ACTIONS:
        raise ValueError(f"unknown universal action: {name}")
    features = [0.0] * ACTION_FEATURE_SIZE
    features[7] = features[8] = features[9] = -1.0
    features[26:] = identity_features(UNIVERSAL_PREFIX + name)
    return features


def candidate_rows(binding_rows: list[dict[str, str]], catalog: CompiledMoveCatalog) -> list[dict[str, Any]]:
    """The candidates CSV rows for every character, from the binding rows.

    Catalog moves keep their catalog features; a stance-branch variant takes
    its base move's features with its own identity hash.
    """

    features = {move["stable_id"]: list(move["action_features"]) for move in catalog.moves}
    by_character: dict[str, list[dict[str, str]]] = {}
    for row in binding_rows:
        by_character.setdefault(row["character"], []).append(row)
    result: list[dict[str, Any]] = []
    for character in [row["slug"] for row in catalog.characters]:
        entries: list[tuple[str, str, str, list[float]]] = [
            (UNIVERSAL_PREFIX + name, "universal", "", universal_action_features(name))
            for name in FULL_MOVE_UNIVERSAL_ACTIONS]
        for row in by_character.get(character, []):
            if int(row["reactive"]):
                continue  # parry outcomes are played by the engine, never chosen
            if row["variant_of"]:
                vector = list(features[row["variant_of"]])
                vector[26:] = identity_features(row["stable_id"])
                entries.append((row["stable_id"], "variant", row["command"], vector))
            else:
                entries.append((row["stable_id"], "move", row["command"], features[row["stable_id"]]))
        if len(entries) > MAX_CANDIDATES:
            raise ValueError(f"{character} has {len(entries)} candidates, above the {MAX_CANDIDATES} bound")
        for index, (stable_id, kind, command, vector) in enumerate(entries):
            result.append({"character": character, "candidate": index, "stable_id": stable_id, "kind": kind,
                           "command": command,
                           **{f"feature_{i}": f"{value:.9g}" for i, value in enumerate(vector)}})
    return result


def build_candidate_table(catalog: CompiledMoveCatalog, character: str | int,
                          candidates_path: str | Path | None = None) -> CandidateTable:
    row = next((value for value in catalog.characters
                if value["slug"] == character or value["id"] == character), None)
    if row is None:
        raise ValueError(f"unknown released character: {character}")
    slug = str(row["slug"])
    path = Path(candidates_path) if candidates_path else _default_candidates_path()
    with path.open(encoding="utf-8") as handle:
        rows = [entry for entry in csv.DictReader(handle) if entry["character"] == slug]
    if [int(entry["candidate"]) for entry in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: {slug} candidates are not numbered 0..n-1")
    universal = [UNIVERSAL_PREFIX + name for name in FULL_MOVE_UNIVERSAL_ACTIONS]
    if [entry["stable_id"] for entry in rows[:len(universal)]] != universal:
        raise ValueError(f"{path}: {slug} does not start with the {len(universal)} universal actions")
    catalog_moves = {move["stable_id"]: move for move in catalog.moves}
    known_states = set(row.get("stances") or [])
    commands: list[dict[str, Any] | None] = []
    for entry in rows:
        if entry["kind"] == "universal":
            commands.append(None)
        elif entry["kind"] == "move":
            commands.append(catalog_moves[entry["stable_id"]]["command"])
        else:
            commands.append(parse_command(entry["command"], known_states).to_dict())
    features = np.asarray([[float(entry[f"feature_{index}"]) for index in range(ACTION_FEATURE_SIZE)]
                           for entry in rows], dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{slug}: invalid candidate features")
    return CandidateTable(slug, tuple(entry["stable_id"] for entry in rows),
                          tuple(entry["kind"] for entry in rows), tuple(commands), features)


def score_candidates(query: np.ndarray, table: CandidateTable) -> np.ndarray:
    """CPU mirror of the CUDA parametric policy head."""

    query = np.asarray(query, dtype=np.float32)
    if query.shape != (ACTION_FEATURE_SIZE,):
        raise ValueError(f"query must have shape ({ACTION_FEATURE_SIZE},), got {query.shape}")
    return (table.features @ query) * np.float32(1.0 / np.sqrt(ACTION_FEATURE_SIZE))


def _default_candidates_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "generated" / CANDIDATES_FILE

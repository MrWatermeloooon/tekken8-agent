"""Live runtime pieces for full-move (parametric) checkpoints.

A full-move checkpoint scores one fighter's variable candidate list (the 18
universal actions plus that fighter's compiled moves, see
``t8_agent.moves.candidates``). Before such a policy may drive a controller:

* the checkpoint must use the full-move action contract and the exact catalog
  and roster it was trained against;
* the controlled character must be named by the operator, match the learner
  character recorded in the checkpoint's trainer state, and pass its complete
  character gate (data, measurements, Practice validation, scenarios, routes);
* every candidate the policy may pick must be legal in the *observed* state.
  Anything the screen cannot establish (stance, Heat, Rage, posture, recovery)
  is ``None`` = unknown, and a move whose requirement depends on an unknown is
  masked out instead of being guessed.

Chosen candidates are dispatched by stable move ID to the compiled controller
command for that move, never to a generic attack label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import Any

import numpy as np

from t8_agent.live.v2_policy import V2Checkpoint
from t8_agent.moves.candidates import (
    FULL_MOVE_ACTION_CONTRACT, UNIVERSAL_PREFIX, CandidateTable, build_candidate_table, score_candidates,
)
from t8_agent.moves.catalog import ACTION_FEATURE_SIZE, CompiledMoveCatalog
from t8_agent.moves.legal import UNIVERSAL_ACTIONS
from t8_agent.moves.candidates import FULL_MOVE_UNIVERSAL_ACTIONS
from t8_agent.moves.validation import _practice_passed, evaluate_character_gate
from t8_agent.sim.actions import SimAction


# Trainer state (src/train.cpp write_trainer_state) up to the learner character.
_STATE_HEADER = struct.Struct("<8sIQQdQQQiQ4fQ8f4Q3Bi")
_STATE_MAGIC = b"T8RUNV2\0"
# Full-move trainer state (src/full_move_train.cpp save_state).
_FULL_MOVE_MAGIC = b"T8FMV1\0\0"


@dataclass(frozen=True)
class TrainerIdentity:
    format_version: int
    completed_update: int
    full_roster: bool
    catalog_sha256: str
    roster_version: str
    learner_character: str
    provisional_geometry: bool = False  # trained with stand-in geometry for unmeasured moves


def _read_strings(raw: bytes, offset: int, count: int, path: str | Path) -> tuple[list[str], int]:
    strings: list[str] = []
    for _ in range(count):
        if offset + 8 > len(raw):
            raise ValueError(f"truncated trainer state: {path}")
        (length,) = struct.unpack_from("<Q", raw, offset)
        offset += 8
        if length > 4096 or offset + length > len(raw):
            raise ValueError(f"corrupt trainer state string: {path}")
        strings.append(raw[offset:offset + length].decode("ascii"))
        offset += length
    return strings, offset


def read_trainer_identity(state_path: str | Path) -> TrainerIdentity:
    """Reads the run identity of a ``.t8state`` or full-move ``.t8fmstate`` file."""

    raw = Path(state_path).read_bytes()
    if raw[:8] == _FULL_MOVE_MAGIC:
        if len(raw) < 20:
            raise ValueError(f"truncated trainer state: {state_path}")
        version, update = struct.unpack_from("<IQ", raw, 8)
        # Versions 1-8 share this prefix (magic, version, update, catalog, roster, learners, provisional).
        if not 1 <= version <= 8:
            raise ValueError(f"unsupported full-move trainer state version {version}: {state_path}")
        (catalog_sha256, roster_version, learner), offset = _read_strings(raw, 20, 3, state_path)
        if offset >= len(raw):
            raise ValueError(f"truncated trainer state: {state_path}")
        return TrainerIdentity(version, update, True, catalog_sha256, roster_version, learner, bool(raw[offset]))
    if len(raw) < _STATE_HEADER.size:
        raise ValueError(f"truncated trainer state: {state_path}")
    fields = _STATE_HEADER.unpack_from(raw)
    if fields[0] != _STATE_MAGIC or not 6 <= fields[1] <= 8:
        raise ValueError(f"not a supported trainer state (format 6-8): {state_path}")
    strings, _ = _read_strings(raw, _STATE_HEADER.size, 3, state_path)
    return TrainerIdentity(int(fields[1]), int(fields[2]), bool(fields[-2]), *strings)


def verify_full_move_checkpoint(
    checkpoint_path: str | Path,
    catalog: CompiledMoveCatalog,
    controlled_character: str,
    *,
    validation_root: str | Path,
    allow_unvalidated: bool = False,
) -> tuple[V2Checkpoint, CandidateTable]:
    """Loads a full-move checkpoint only for the character it was trained on.

    ``allow_unvalidated`` skips the character gate; use it only with
    controller output disabled.
    """

    if not controlled_character:
        raise ValueError("a full-move checkpoint requires the controlled character to be named")
    checkpoint_path = Path(checkpoint_path)
    checkpoint = V2Checkpoint.load(checkpoint_path)
    if checkpoint.action_contract != FULL_MOVE_ACTION_CONTRACT:
        raise ValueError(f"checkpoint action contract is {checkpoint.action_contract!r}, "
                         f"not {FULL_MOVE_ACTION_CONTRACT!r}")
    if checkpoint.action_feature_size != ACTION_FEATURE_SIZE:
        raise ValueError(f"checkpoint scores {checkpoint.action_feature_size}-feature candidates; "
                         f"the catalog has {ACTION_FEATURE_SIZE}")
    if checkpoint.universal_action_count != len(FULL_MOVE_UNIVERSAL_ACTIONS):
        raise ValueError("checkpoint universal-action count does not match this runtime")
    if checkpoint.catalog_sha256 != catalog.catalog_sha256:
        raise ValueError("checkpoint was trained against a different move catalog "
                         f"({checkpoint.catalog_sha256[:12]} vs {catalog.catalog_sha256[:12]})")
    roster_version = str(catalog.document.get("roster_as_of", ""))
    if checkpoint.roster_version != roster_version:
        raise ValueError(f"checkpoint roster {checkpoint.roster_version!r} differs from {roster_version!r}")
    state_path = checkpoint_path.with_suffix(".t8fmstate")
    if not state_path.exists():
        state_path = checkpoint_path.with_suffix(".t8state")
    if not state_path.exists():
        raise ValueError(f"cannot verify the trained character: missing {checkpoint_path.stem}.t8fmstate")
    identity = read_trainer_identity(state_path)
    # Only --screen-observation checkpoints (full-move-screen-*) trained on what the screen shows.
    # Every other full-move contract (privileged, memory, noise) reads simulator state.
    contract = checkpoint.observation_contract
    if contract.startswith("full-move-") and not contract.startswith("full-move-screen") and not allow_unvalidated:
        raise ValueError(f"checkpoint observes {checkpoint.observation_contract!r}, the simulator's full state; "
                         "live play needs a screen-observable full-move contract (--screen-observation)")
    if identity.provisional_geometry and not allow_unvalidated:
        raise ValueError("checkpoint was trained with stand-in geometry for unmeasured moves; "
                         "it is a development checkpoint, not one for controller output")
    # Full-move runs record the comma-separated characters one policy learned to play.
    trained = identity.learner_character.split(",")
    if controlled_character not in trained:
        raise ValueError(f"checkpoint was trained for {identity.learner_character!r}, "
                         f"but the controlled character is {controlled_character!r}")
    if identity.catalog_sha256 != catalog.catalog_sha256:
        raise ValueError("checkpoint trainer state names a different move catalog")
    if not allow_unvalidated:
        report = evaluate_character_gate(catalog, controlled_character, validation_root)
        if not report.ready:
            raise ValueError(f"{controlled_character} has not passed its character gate: "
                             + "; ".join(report.blockers))
    return checkpoint, build_candidate_table(catalog, controlled_character)


@dataclass(frozen=True)
class ObservedMoveState:
    """What the live runtime knows about its own fighter. ``None`` = unknown."""

    busy: bool | None = None
    posture: str | None = None          # standing, crouching, while_standing
    stance: str | None = None           # "" = no stance
    heat: bool | None = None
    rage: bool | None = None
    airborne: bool | None = None
    knocked_down: bool | None = None
    situations: frozenset[str] = field(default_factory=frozenset)  # observed, e.g. OPPONENT_WALL_STUN

    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(name for name in ("busy", "posture", "stance", "heat", "rage", "airborne", "knocked_down")
                     if getattr(self, name) is None)


@dataclass(frozen=True)
class OpponentMoveObservation:
    """The opponent's current move as far as the screen shows it.

    No validated move-family recognizer exists yet, so the runtime reports
    ``family="unknown"`` with the raw activity cues rather than a guessed label.
    """

    family: str = "unknown"
    confidence: float = 0.0
    active: bool = False
    attack_cue: float = 0.0


def live_legal_mask(
    table: CandidateTable,
    catalog: CompiledMoveCatalog,
    observed: ObservedMoveState,
    *,
    ledger: dict[str, Any] | None = None,
    require_validated: bool = True,
) -> np.ndarray:
    """Candidate mask that never allows a move whose requirements are unknown."""

    mask = np.zeros(len(table), dtype=bool)
    mask[0] = True  # neutral is always sendable
    if observed.busy is not False:
        return mask
    downed = observed.airborne is not False or observed.knocked_down is not False
    if not downed:
        # Heat Dash, wake-ups, and tech rolls have no validated controller input yet.
        mask[1:len(UNIVERSAL_ACTIONS)] = True
    moves_by_id = {move["stable_id"]: move for move in catalog.moves}
    practice = (ledger or {}).get("moves") or {}
    for index in range(table.universal_count, len(table)):
        stable_id = table.stable_ids[index]
        # A stance-branch variant ("jun:67~GEN") is legal where its base move is.
        move = moves_by_id[stable_id.split("~", 1)[0]]
        command = table.commands[index] or {}
        mask[index] = (not downed and command.get("parser_status") == "parsed" and
                       _move_legal(move, observed) and
                       (not require_validated or (_practice_passed(practice.get(stable_id)) and
                                                  move["validation"]["simulation"] == "ready")))
    return mask


def _move_legal(move: dict[str, Any], observed: ObservedMoveState) -> bool:
    command = move["command"]
    if command["parser_status"] != "parsed" or move["validation"]["source_consistency"] != "valid":
        return False
    rule = move["legal_state"]
    if rule["posture"] != "standing" and observed.posture != rule["posture"]:
        return False
    if rule["posture"] == "standing" and observed.posture not in ("standing", "while_standing"):
        return False
    if rule["requires_heat"] and observed.heat is not True:
        return False
    if rule["requires_rage"] and observed.rage is not True:
        return False
    if rule["stances"]:
        if observed.stance is None or observed.stance.upper() not in rule["stances"]:
            return False
    elif observed.stance is None or observed.stance:
        return False  # from an unknown or other stance the same input is a different move
    if not set(rule["situations"]) <= observed.situations:
        return False
    # Mid-string requirements (a stance or outcome later in the string) must
    # also be observable before the move can be planned.
    step_requirements = {str(step.get("requirement") or "").upper() for step in command["steps"]} - {"", "WS", "SS"}
    return step_requirements <= observed.situations


class FullMovePolicy:
    """CPU inference for a parametric checkpoint (mirrors ppo.cu)."""

    def __init__(self, checkpoint: V2Checkpoint, table: CandidateTable, *, deterministic: bool = True,
                 seed: int = 2027) -> None:
        if checkpoint.action_feature_size != table.features.shape[1]:
            raise ValueError("checkpoint and candidate table feature sizes differ")
        self.checkpoint = checkpoint
        self.table = table
        self.deterministic = deterministic
        self._rng = np.random.default_rng(seed)

    def scores(self, observation: np.ndarray) -> np.ndarray:
        checkpoint = self.checkpoint
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (checkpoint.observation_size,) or not np.isfinite(observation).all():
            raise ValueError(f"observation must be finite with shape ({checkpoint.observation_size},)")
        vector = checkpoint.normalize(observation)
        hidden = np.tanh(checkpoint.weights_1 @ vector + checkpoint.bias_1)
        hidden = np.tanh(checkpoint.weights_2 @ hidden + checkpoint.bias_2)
        output = checkpoint.weights_out @ hidden + checkpoint.bias_out
        return score_candidates(output[:checkpoint.action_feature_size], self.table)

    def act(self, observation: np.ndarray, mask: np.ndarray) -> int:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self.table),) or not mask.any():
            raise ValueError("mask must cover every candidate and allow at least one")
        scores = np.where(mask, self.scores(observation), -np.inf)
        if self.deterministic:
            return int(np.argmax(scores))
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        return int(self._rng.choice(len(scores), p=probabilities))


def dispatch_candidate(controller: Any, table: CandidateTable, index: int, *,
                       expected_generation: int | None = None) -> bool:
    """Sends a chosen candidate by its stable ID: a universal action or the
    move's compiled command. Raises for a candidate that cannot be sent."""

    stable_id = table.stable_ids[index]
    if table.is_universal(index):
        name = stable_id[len(UNIVERSAL_PREFIX):]
        if name not in UNIVERSAL_ACTIONS:
            raise ValueError(f"{stable_id} has no validated controller input")
        action = SimAction(name)
        return bool(controller.send_action(action, expected_generation=expected_generation))
    command = table.commands[index]
    if command is None:
        raise ValueError(f"{stable_id} has no compiled command")
    return bool(controller.send_command(command, move_id=stable_id, expected_generation=expected_generation))


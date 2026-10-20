from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.live.feedback import (
    FAILED, HIT, INTERRUPTED, NO_HIT, NOT_ACCEPTED, ExecutionFeedback, ReplayRecorder, SideSwitchDetector,
    compare_replay,
)
from t8_agent.live.full_move import (
    FullMovePolicy, ObservedMoveState, dispatch_candidate, live_legal_mask, read_trainer_identity,
    verify_full_move_checkpoint,
)
from t8_agent.live.v2_policy import V2Checkpoint, _fnv1a
from t8_agent.moves.candidates import (
    FULL_MOVE_ACTION_CONTRACT, FULL_MOVE_UNIVERSAL_ACTIONS, MAX_CANDIDATES, build_candidate_table, score_candidates,
)
from t8_agent.moves.catalog import ACTION_FEATURE_SIZE, compile_catalog
from t8_agent.moves.legal import UNIVERSAL_ACTIONS
from t8_agent.vision.temporal import VisualEstimate


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog():
    return compile_catalog(REPO_ROOT / "data")


def write_full_move_checkpoint(path: Path, catalog, *, query_weights: np.ndarray | None = None,
                               action_contract: bytes = FULL_MOVE_ACTION_CONTRACT.encode()) -> None:
    observations, hidden, features = 13, 4, ACTION_FEATURE_SIZE
    output = features + 1
    weights_1 = np.zeros((hidden, observations), "<f4")
    weights_1[0, 0] = 1.0                       # hidden 0 follows own health
    weights_2 = np.eye(hidden, dtype="<f4")
    weights_out = np.zeros((output, hidden), "<f4")
    if query_weights is not None:
        weights_out[:features, 0] = query_weights
    tensors = [weights_1, np.zeros(hidden, "<f4"), weights_2, np.zeros(hidden, "<f4"),
               weights_out, np.zeros(output, "<f4")]
    model = b"".join(tensor.tobytes() for tensor in tensors)
    payload = model + bytes(len(model) * 2)
    header = struct.pack("<8sIIIIQQ", b"T8V2PPO\0", 3, observations, MAX_CANDIDATES, hidden, 5,
                         len(model) // 4)
    contract = struct.pack("<64s24s32s32sII", catalog.catalog_sha256.encode(),
                           str(catalog.document["roster_as_of"]).encode(), b"visual-13-v2",
                           action_contract, features, len(FULL_MOVE_UNIVERSAL_ACTIONS))
    path.write_bytes(header + contract + struct.pack("<QQ", len(payload), _fnv1a(payload)) + payload)


def write_trainer_state(path: Path, *, learner: str, catalog_sha256: str, roster: str = "2026-07-21") -> None:
    header = struct.pack("<8sIQQdQQQiQ4fQ8f4Q3Bi", b"T8RUNV2\0", 8, 100, 4096, 1.5, 2027, 64, 128, 4, 64,
                         *([0.1] * 4), 100, *([0.1] * 8), 100, 10, 10, 16, 0, 1, 1, 0)

    def string(value: str) -> bytes:
        return struct.pack("<Q", len(value)) + value.encode()

    path.write_bytes(header + string(catalog_sha256) + string(roster) + string(learner) + bytes(16))


def estimate(**overrides) -> VisualEstimate:
    values = dict(p1_health_ratio=1.0, p2_health_ratio=1.0, p1_x=-0.5, p2_x=0.5, distance=1.0,
                  p1_velocity=0.0, p2_velocity=0.0, p1_motion=0.0, p2_motion=0.0, p1_hit_event=False,
                  p2_hit_event=False, p1_attack_likelihood=0.0, p2_attack_likelihood=0.0)
    values.update(overrides)
    return VisualEstimate(**values)


class FakeGamepad:
    def __init__(self) -> None:
        self.held: set[str] = set()

    def press_button(self, *, button: str) -> None:
        self.held.add(button)

    def release_button(self, *, button: str) -> None:
        self.held.discard(button)

    def update(self) -> None:
        pass


def test_candidate_table_orders_universal_actions_then_moves(catalog):
    table = build_candidate_table(catalog, "jun")
    universal = len(FULL_MOVE_UNIVERSAL_ACTIONS)
    assert table.universal_count == universal == len(UNIVERSAL_ACTIONS) + 4
    assert table.stable_ids[0] == "universal:neutral" and table.commands[0] is None
    assert table.stable_ids[universal - 1] == "universal:tech_roll"
    # Every non-reactive Jun move follows, then stance-branch variants in binding order.
    moves = [move for move in catalog.moves if move["stable_id"].startswith("jun:")]
    reactive = {"jun:3", "jun:142", "jun:143", "jun:147"}
    assert set(table.stable_ids) >= {move["stable_id"] for move in moves} - reactive
    assert table.kinds.count("variant") == 15 and "jun:67~GEN" in table.stable_ids
    assert all(kind == "universal" for kind in table.kinds[:universal])
    assert table.features.shape == (len(table), ACTION_FEATURE_SIZE) and table.features.dtype == np.float32
    # Universal vectors are distinct from each other and from every move.
    assert len({row.tobytes() for row in table.features}) == len(table)
    largest = max(len(build_candidate_table(catalog, row["slug"])) for row in catalog.characters)
    assert largest <= MAX_CANDIDATES


def test_cpu_scorer_matches_the_cuda_formula(catalog):
    table = build_candidate_table(catalog, "jun")
    query = np.linspace(-1, 1, ACTION_FEATURE_SIZE, dtype=np.float32)
    expected = np.asarray([float(np.dot(query, row)) / np.sqrt(ACTION_FEATURE_SIZE) for row in table.features])
    np.testing.assert_allclose(score_candidates(query, table), expected, rtol=1e-5, atol=1e-6)


def test_trainer_identity_reads_the_learner_character(tmp_path):
    path = tmp_path / "update_100.t8state"
    write_trainer_state(path, learner="jun", catalog_sha256="a" * 64)
    identity = read_trainer_identity(path)
    assert (identity.learner_character, identity.catalog_sha256, identity.completed_update) == ("jun", "a" * 64, 100)
    path.write_bytes(b"NOTSTATE" + bytes(200))
    with pytest.raises(ValueError, match="not a supported trainer state"):
        read_trainer_identity(path)


def test_full_move_checkpoint_requires_matching_character_contract_and_gate(tmp_path, catalog):
    checkpoint = tmp_path / "update_100.t8ppo"
    write_full_move_checkpoint(checkpoint, catalog)
    with pytest.raises(ValueError, match="missing update_100.t8fmstate"):
        verify_full_move_checkpoint(checkpoint, catalog, "jun", validation_root=tmp_path, allow_unvalidated=True)
    write_trainer_state(checkpoint.with_suffix(".t8state"), learner="jun", catalog_sha256=catalog.catalog_sha256)
    with pytest.raises(ValueError, match="trained for 'jun'"):
        verify_full_move_checkpoint(checkpoint, catalog, "reina", validation_root=tmp_path, allow_unvalidated=True)
    with pytest.raises(ValueError, match="requires the controlled character"):
        verify_full_move_checkpoint(checkpoint, catalog, "", validation_root=tmp_path)
    with pytest.raises(ValueError, match="has not passed its character gate"):
        verify_full_move_checkpoint(checkpoint, catalog, "jun", validation_root=tmp_path)
    loaded, table = verify_full_move_checkpoint(checkpoint, catalog, "jun", validation_root=tmp_path,
                                                allow_unvalidated=True)
    assert table.character == "jun" and loaded.action_feature_size == ACTION_FEATURE_SIZE

    compat = tmp_path / "compat.t8ppo"
    write_full_move_checkpoint(compat, catalog, action_contract=b"compatibility-six-v3")
    with pytest.raises(ValueError, match="action contract"):
        verify_full_move_checkpoint(compat, catalog, "jun", validation_root=tmp_path, allow_unvalidated=True)


def test_live_mask_rejects_moves_that_depend_on_unknown_state(catalog):
    table = build_candidate_table(catalog, "jun")
    unknown = live_legal_mask(table, catalog, ObservedMoveState(busy=False), require_validated=False)
    assert unknown[0] and unknown.sum() == 1, "unknown airborne/knockdown state allows only neutral"
    grounded = ObservedMoveState(busy=False, airborne=False, knocked_down=False)
    assert not live_legal_mask(table, catalog, grounded, require_validated=False)[len(FULL_MOVE_UNIVERSAL_ACTIONS):].any(), \
        "unknown stance and posture allow no move"
    known = ObservedMoveState(busy=False, airborne=False, knocked_down=False, posture="standing", stance="",
                              heat=False, rage=False)
    mask = live_legal_mask(table, catalog, known, require_validated=False)
    moves = {move["stable_id"]: move for move in catalog.moves}
    legal_ids = [table.stable_ids[index] for index in np.flatnonzero(mask[len(FULL_MOVE_UNIVERSAL_ACTIONS):]) +
                 len(FULL_MOVE_UNIVERSAL_ACTIONS)]
    assert legal_ids, "standard standing moves are legal once the state is known"
    for stable_id in legal_ids:
        rule = moves[stable_id.split("~", 1)[0]]["legal_state"]
        assert not rule["requires_heat"] and not rule["stances"] and not rule["situations"]
    # The 18 live universal actions are sendable; Heat Dash, wake-ups, and tech rolls are not yet.
    assert mask[:len(UNIVERSAL_ACTIONS)].all() and not mask[len(UNIVERSAL_ACTIONS):len(FULL_MOVE_UNIVERSAL_ACTIONS)].any()
    # Strict mode also needs a Practice pass with provenance for each move.
    assert not live_legal_mask(table, catalog, known)[len(FULL_MOVE_UNIVERSAL_ACTIONS):].any()
    assert live_legal_mask(table, catalog, ObservedMoveState(busy=True)).sum() == 1


def test_full_move_policy_picks_and_dispatches_by_stable_id(catalog):
    table = build_candidate_table(catalog, "jun")
    target = next(index for index in range(len(FULL_MOVE_UNIVERSAL_ACTIONS), len(table))
                  if table.commands[index]["parser_status"] == "parsed" and not table.commands[index]["requirements"]
                  and not any(step["requirement"] for step in table.commands[index]["steps"]))
    query = table.features[target] * 50.0
    checkpoint = V2Checkpoint(
        observation_size=13, action_count=MAX_CANDIDATES, hidden_size=1, optimizer_step=0,
        catalog_sha256=catalog.catalog_sha256, roster_version="2026-07-21", observation_contract="visual-13-v2",
        action_contract=FULL_MOVE_ACTION_CONTRACT, action_feature_size=ACTION_FEATURE_SIZE,
        universal_action_count=len(FULL_MOVE_UNIVERSAL_ACTIONS),
        weights_1=np.zeros((1, 13), np.float32), bias_1=np.full(1, 10.0, np.float32),
        weights_2=np.zeros((1, 1), np.float32), bias_2=np.full(1, 10.0, np.float32),
        weights_out=np.concatenate([query, [0.0]]).astype(np.float32).reshape(-1, 1),
        bias_out=np.zeros(ACTION_FEATURE_SIZE + 1, np.float32))
    policy = FullMovePolicy(checkpoint, table)
    mask = np.ones(len(table), dtype=bool)
    assert policy.act(np.zeros(13, np.float32), mask) == target
    mask[target] = False
    assert policy.act(np.zeros(13, np.float32), mask) != target

    sent = []

    class Recorder:
        def send_action(self, action, *, expected_generation=None):
            sent.append(("action", action.value))
            return True

        def send_command(self, command, *, move_id, expected_generation=None):
            sent.append(("command", move_id, command["raw"]))
            return True

    assert dispatch_candidate(Recorder(), table, target)
    assert dispatch_candidate(Recorder(), table, 3)
    assert sent == [("command", table.stable_ids[target], table.commands[target]["raw"]),
                    ("action", "dash_forward")]


def test_execution_feedback_classifies_outcomes():
    feedback = ExecutionFeedback(tolerance_frames=2, active_frames=4)
    startup = {"min": 10, "max": 12}
    attempt = feedback.start("jun:1", sent=True, startup=startup, now=0.0)
    feedback.update(estimate(p1_motion=0.05), now=2 / 60)
    assert feedback.update(estimate(p2_hit_event=True), now=11 / 60).outcome == HIT
    assert attempt.accepted_at is not None

    feedback.start("jun:2", sent=True, startup=startup, now=1.0)
    assert feedback.update(estimate(p1_hit_event=True), now=1.0 + 5 / 60).outcome == INTERRUPTED

    feedback.start("jun:3", sent=True, startup=startup, now=2.0)
    assert feedback.update(estimate(), now=2.0 + 15 / 60).outcome == NOT_ACCEPTED

    feedback.start("jun:4", sent=True, startup=startup, now=3.0)
    feedback.update(estimate(p1_motion=0.05), now=3.0 + 1 / 60)
    assert feedback.update(estimate(), now=3.0 + 15 / 60) is None
    assert feedback.update(estimate(), now=3.0 + 19 / 60).outcome == NO_HIT

    assert feedback.start("jun:5", sent=False, now=4.0).outcome == FAILED
    assert feedback.summary() == {HIT: 1, INTERRUPTED: 1, NOT_ACCEPTED: 1, NO_HIT: 1, FAILED: 1}


def test_side_switch_detector_needs_a_sustained_color_swap():
    red = np.zeros((20, 20, 3), np.uint8)
    red[..., 0] = 200
    blue = np.zeros((20, 20, 3), np.uint8)
    blue[..., 2] = 200
    detector = SideSwitchDetector(confirm_frames=3)
    assert not detector.update(red, blue)        # reference
    assert not detector.update(red, blue)
    assert not detector.update(blue, red) and not detector.update(blue, red)
    assert not detector.update(red, blue), "a brief swap (camera cut) resets the count"
    results = [detector.update(blue, red) for _ in range(3)]
    assert results == [False, False, True]
    assert not detector.update(blue, red), "the swapped colors become the new reference"
    assert not detector.update(np.zeros((20, 20, 3), np.uint8), red), "dark crops carry no signature"


def test_replay_compares_expected_and_recorded_inputs(tmp_path):
    path = tmp_path / "replay.jsonl"
    recorder = ReplayRecorder(path, metadata={"agent": "test"})
    controller = VGamepadInputBackend(gamepad=FakeGamepad(), tap_seconds=0.0, dash_gap_seconds=0.0,
                                      frame_seconds=0.0, input_log=recorder.controller)
    controller.set_enabled(True)
    from t8_agent.moves.notation import parse_command
    command = parse_command("df+1")
    planned = controller.planned_inputs(command=command, move_id="jun:x")
    assert planned[-2] == frozenset() and frozenset({"down", "right", "x"}) in planned
    recorder.decision(estimate=estimate(), candidate="jun:x", sent=True, legal_count=3,
                      expected_inputs=planned, time_value=0.0)
    controller.send_command(command, move_id="jun:x")
    recorder.decision(estimate=estimate(), candidate="jun:y", sent=True, legal_count=3,
                      expected_inputs=[frozenset({"y"})], time_value=1e12)
    feedback = ExecutionFeedback()
    attempt = feedback.start("jun:x", sent=False, now=0.0)
    recorder.outcome(attempt)
    recorder.close()
    report = compare_replay(path)
    assert report["decisions"] == 2 and report["sent"] == 2
    assert [mismatch["candidate"] for mismatch in report["input_mismatches"]] == ["jun:y"]
    assert report["outcomes"] == {FAILED: 1}
    assert all(json.loads(line)["kind"] for line in path.read_text(encoding="utf-8").splitlines())


def test_full_move_trainer_state_identity_and_provisional_refusal(tmp_path, catalog):
    def string(value: str) -> bytes:
        return struct.pack("<Q", len(value)) + value.encode()

    checkpoint = tmp_path / "update_8.t8ppo"
    write_full_move_checkpoint(checkpoint, catalog)
    state = (b"T8FMV1" + bytes(2) + struct.pack("<IQ", 1, 8) + string(catalog.catalog_sha256) +
             string("2026-07-21") + string("jun") + bytes([1]) + string("identity"))
    checkpoint.with_suffix(".t8fmstate").write_bytes(state)
    identity = read_trainer_identity(checkpoint.with_suffix(".t8fmstate"))
    assert (identity.learner_character, identity.completed_update, identity.provisional_geometry) == ("jun", 8, True)
    with pytest.raises(ValueError, match="stand-in geometry"):
        verify_full_move_checkpoint(checkpoint, catalog, "jun", validation_root=tmp_path)
    _, table = verify_full_move_checkpoint(checkpoint, catalog, "jun", validation_root=tmp_path,
                                           allow_unvalidated=True)
    assert table.character == "jun"

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from t8_agent.live.v2_policy import LiveV2GpuAgent, V2Checkpoint, _fnv1a
from t8_agent.roster.temporal import TemporalFrame
from t8_agent.sim.action_space import ACTION_SPACE
from t8_agent.vision.temporal import VisualEstimate


def write_checkpoint(path: Path, observations: int = 13, contract_name: bytes | None = None) -> None:
    actions = len(ACTION_SPACE)
    hidden = 4
    output = actions + 1
    model_count = hidden * observations + hidden + hidden * hidden + hidden + output * hidden + output
    tensors = [
        np.zeros((hidden, observations), dtype="<f4"),
        np.zeros(hidden, dtype="<f4"),
        np.zeros((hidden, hidden), dtype="<f4"),
        np.zeros(hidden, dtype="<f4"),
        np.zeros((output, hidden), dtype="<f4"),
        np.asarray([0.0, 1.0, 2.0, 9.0] + [0.0] * (output - 4), dtype="<f4"),
    ]
    model = b"".join(tensor.tobytes() for tensor in tensors)
    payload = model + bytes(len(model) * 2)
    header = struct.pack("<8sIIIIQQ", b"T8V2PPO\0", 3, observations, actions, hidden, 17, model_count)
    contract = struct.pack(
        "<64s24s32s32sII",
        b"0" * 64,
        b"compatibility",
        contract_name or (b"visual-13-v2" if observations == 13 else b"visual-matchup-95-v2"),
        b"fixed-24-compatibility",
        0,
        18,
    )
    integrity = struct.pack("<QQ", len(payload), _fnv1a(payload))
    path.write_bytes(header + contract + integrity + payload)


def write_normalized_checkpoint(path: Path, *, enabled: int, count: int,
                                mean: np.ndarray, variance: np.ndarray) -> None:
    """V4 checkpoint whose action 5 vs 6 choice follows the sign of normalized feature 2."""
    observations, actions, hidden = 13, len(ACTION_SPACE), 4
    output = actions + 1
    weights_1 = np.zeros((hidden, observations), dtype="<f4")
    weights_1[0, 2] = 1.0
    weights_2 = np.zeros((hidden, hidden), dtype="<f4")
    weights_2[0, 0] = 1.0
    weights_out = np.zeros((output, hidden), dtype="<f4")
    weights_out[5, 0], weights_out[6, 0] = 5.0, -5.0
    tensors = [weights_1, np.zeros(hidden, "<f4"), weights_2, np.zeros(hidden, "<f4"),
               weights_out, np.zeros(output, "<f4")]
    model = b"".join(tensor.tobytes() for tensor in tensors)
    model_count = len(model) // 4
    payload = (model + bytes(len(model) * 2) + np.asarray(mean, "<f4").tobytes() +
               np.asarray(variance, "<f4").tobytes())
    header = struct.pack("<8sIIIIQQ", b"T8V2PPO\0", 4, observations, actions, hidden, 17, model_count)
    contract = struct.pack("<64s24s32s32sII", b"0" * 64, b"compatibility", b"visual-13-v2",
                           b"fixed-24-compatibility", 0, 18)
    normalizer = struct.pack("<IIQff", enabled, 0, count, 10.0, 1e-8)
    integrity = struct.pack("<QQ", len(payload), _fnv1a(payload))
    path.write_bytes(header + contract + normalizer + integrity + payload)


def test_v4_checkpoint_applies_observation_normalization(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    mean = np.zeros(13, dtype=np.float32)
    mean[2] = -2.0  # estimate() own_x = -0.85 normalizes to +1.15
    variance = np.ones(13, dtype=np.float32)
    normalized_path = tmp_path / "normalized.t8ppo"
    write_normalized_checkpoint(normalized_path, enabled=1, count=100, mean=mean, variance=variance)
    loaded = V2Checkpoint.load(normalized_path)
    assert loaded.observation_mean is not None
    assert loaded.normalize(estimate().to_vector().astype(np.float32))[2] == pytest.approx(1.15)
    assert LiveV2GpuAgent(normalized_path, device="cpu").act(estimate()) == ACTION_SPACE[5]

    # Disabled (or never-updated) normalizers pass raw inputs, as in training.
    for enabled, count in ((0, 100), (1, 0)):
        raw_path = tmp_path / f"raw_{enabled}_{count}.t8ppo"
        write_normalized_checkpoint(raw_path, enabled=enabled, count=count, mean=mean, variance=variance)
        assert V2Checkpoint.load(raw_path).observation_mean is None
        assert LiveV2GpuAgent(raw_path, device="cpu").act(estimate()) == ACTION_SPACE[6]


def test_v4_normalizer_clips_and_rejects_invalid_statistics(tmp_path: Path) -> None:
    path = tmp_path / "clip.t8ppo"
    mean = np.full(13, 1000.0, dtype=np.float32)
    write_normalized_checkpoint(path, enabled=1, count=5, mean=mean, variance=np.ones(13, np.float32))
    normalized = V2Checkpoint.load(path).normalize(np.zeros(13, dtype=np.float32))
    assert normalized.tolist() == [-10.0] * 13

    bad_path = tmp_path / "negative_variance.t8ppo"
    write_normalized_checkpoint(bad_path, enabled=1, count=5, mean=mean,
                                variance=np.full(13, -1.0, np.float32))
    with pytest.raises(ValueError, match="observation statistics"):
        V2Checkpoint.load(bad_path)


def estimate() -> VisualEstimate:
    return VisualEstimate(1.0, 1.0, -0.85, 0.85, 1.7, 0.0, 0.0, 0.0, 0.0,
                          False, False, 0.0, 0.0)


def test_native_checkpoint_load_and_inference(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "visual.t8ppo"
    write_checkpoint(checkpoint_path)
    loaded = V2Checkpoint.load(checkpoint_path)
    assert loaded.observation_size == 13
    assert loaded.action_count == len(ACTION_SPACE)
    assert loaded.weights_1.shape == (4, 13)
    torch = pytest.importorskip("torch")
    del torch
    agent = LiveV2GpuAgent(checkpoint_path, device="cpu", deterministic=True)
    assert agent.act(estimate()) == ACTION_SPACE[3]
    p2_agent = LiveV2GpuAgent(checkpoint_path, device="cpu", deterministic=True, player=2)
    asymmetric = VisualEstimate(0.9, 0.4, -0.7, 1.1, 1.8, 0.1, -0.2, 0.3, 0.6,
                                False, True, 0.2, 0.8)
    assert p2_agent.observation(asymmetric).tolist() == pytest.approx(
        [0.4, 0.9, 1.1, -0.7, 1.8, -0.2, 0.1, 0.6, 0.3, 1.0, 0.0, 0.8, 0.2]
    )


def test_native_checkpoint_cuda_inference(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available")
    checkpoint_path = tmp_path / "visual_cuda.t8ppo"
    write_checkpoint(checkpoint_path)
    agent = LiveV2GpuAgent(checkpoint_path, device="cuda", deterministic=True)
    logits = agent._logits_tensor(agent.observation(estimate()))
    assert agent.weights_1.is_cuda
    assert logits.is_cuda
    assert torch.isfinite(logits).all()
    assert agent.act(estimate()) == ACTION_SPACE[3]


def test_matchup_checkpoint_requires_context_and_runs_95_features(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    checkpoint_path = tmp_path / "visual_matchup.t8ppo"
    write_checkpoint(checkpoint_path, observations=95)
    with pytest.raises(ValueError, match="requires opponent_character"):
        LiveV2GpuAgent(checkpoint_path, device="cpu")
    agent = LiveV2GpuAgent(
        checkpoint_path,
        device="cpu",
        opponent_character="reina",
        opponent_archetype="movement_specialist",
    )
    observation = agent.observation(
        estimate(),
        temporal_frame=TemporalFrame(
            move_id=18, animation_phase=0.5, hit_level=1, distance=1.7,
        ),
    )
    assert observation.shape == (95,)
    assert observation[21 + 8] == 1.0
    assert np.count_nonzero(observation[-8:]) >= 4
    assert agent.act(estimate()) == ACTION_SPACE[3]
    agent.reset_episode()


def test_native_checkpoint_rejects_corruption(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "corrupt.t8ppo"
    write_checkpoint(checkpoint_path)
    raw = bytearray(checkpoint_path.read_bytes())
    raw[-1] ^= 0x40
    checkpoint_path.write_bytes(raw)
    with pytest.raises(ValueError, match="integrity"):
        V2Checkpoint.load(checkpoint_path)


def test_native_checkpoint_rejects_legacy_fixed_action_version(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy.t8ppo"
    write_checkpoint(checkpoint_path)
    raw = bytearray(checkpoint_path.read_bytes())
    struct.pack_into("<I", raw, 8, 2)
    checkpoint_path.write_bytes(raw)
    with pytest.raises(ValueError, match="legacy fixed-action checkpoint"):
        V2Checkpoint.load(checkpoint_path)


def test_live_policy_masks_logits_before_argmax_and_sampling(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    path = tmp_path / "masked.t8ppo"
    write_checkpoint(path)
    agent = LiveV2GpuAgent(path, device="cpu")
    mask = np.zeros(len(ACTION_SPACE), dtype=bool)
    mask[12] = True
    for deterministic in (True, False):
        agent.deterministic = deterministic
        assert agent.act(estimate(), action_mask=mask) == ACTION_SPACE[12]
    with pytest.raises(ValueError, match="action mask"):
        agent.act(estimate(), action_mask=np.zeros(len(ACTION_SPACE), dtype=bool))


def test_unidentified_motion_does_not_claim_jab_or_high_hit(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from dataclasses import replace

    path = tmp_path / "unknown.t8ppo"
    write_checkpoint(path, observations=95)
    agent = LiveV2GpuAgent(path, device="cpu", opponent_character="reina", opponent_archetype="rushdown")
    observation = agent.observation(replace(estimate(), p2_attack_likelihood=0.9))
    assert observation[-8:-3].tolist() == [-1.0] * 5
    agent.observation(replace(estimate(), p2_health_ratio=0.5))
    agent.reset_episode()
    reset = agent.observation(estimate())
    assert not reset[31:-8].any()
    assert reset[-3] == 0.0


def test_screen_checkpoint_uses_only_screen_features(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from dataclasses import replace

    path = tmp_path / "screen.t8ppo"
    write_checkpoint(path, observations=95, contract_name=b"screen-matchup-95-v1")
    with pytest.raises(ValueError, match="calibrated"):
        LiveV2GpuAgent(path, device="cpu", opponent_character="reina", opponent_archetype="rushdown")
    agent = LiveV2GpuAgent(path, device="cpu", opponent_character="reina", opponent_archetype="rushdown",
                           screen_position_sigma=0.4, screen_event_error=0.1, motion_threshold=0.015)
    assert agent.screen_only
    moving = replace(estimate(), p2_motion=0.03, p2_attack_likelihood=0.7, p1_motion=0.001)
    first = agent.observation(moving)
    second = agent.observation(moving)
    # Base: motion and attack likelihood become binary detections.
    assert first[7] == 0.0 and first[8] == 1.0 and first[11] == 0.0 and first[12] == 1.0
    newest = second[-8:]
    assert newest[0] == 1.0 and newest[1] == 1.0            # activity, attack cue
    assert newest[2] == pytest.approx(0.4)                    # sigma / 1.0
    assert newest[3] == pytest.approx(0.2)                    # error / 0.5
    assert newest[4] == pytest.approx(8 / 60)                 # two active decisions
    assert newest[6] == pytest.approx(1.7 / 7.2)
    assert not (second[31:-8] < 0).any()                      # no "unknown move" placeholders
    idle = agent.observation(replace(moving, p2_motion=0.0, p2_attack_likelihood=0.0))
    assert idle[-8] == 0.0 and idle[-4] == 0.0                # activity run resets
    agent.reset_episode()
    assert agent.observation(moving)[-4] == pytest.approx(4 / 60)


def test_screen_uncertainty_calibration_math(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "calibrate", Path(__file__).resolve().parents[1] / "scripts" / "calibrate_screen_uncertainty.py")
    calibrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibrate)
    result = calibrate.position_sigma_from_pairs([(2.0, 1.0), (1.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
    assert result["bias"] == pytest.approx(0.0)
    assert result["rms"] == pytest.approx(math.sqrt(0.5))
    assert result["sigma"] == pytest.approx(0.5)
    events = calibrate.event_error_from_labels(
        [(True, False), (False, False), (True, True), (False, True)],
        [(True, False), (True, False), (True, True), (False, False)])
    assert events["error"] == pytest.approx(2 / 8)
    config = tmp_path / "screen.yaml"
    config.write_text("motion_threshold: 0.02\ncapture_fps: 60\n", encoding="utf-8")
    calibrate.update_config(config, {"screen_position_sigma": 0.5})
    import yaml
    assert yaml.safe_load(config.read_text()) == {
        "motion_threshold": 0.02, "capture_fps": 60, "screen_position_sigma": 0.5}

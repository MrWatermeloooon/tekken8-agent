from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from t8_agent.live.v2_policy import LiveV2GpuAgent, V2Checkpoint, _fnv1a
from t8_agent.roster.temporal import TemporalFrame
from t8_agent.sim.action_space import ACTION_SPACE
from t8_agent.vision.temporal import VisualEstimate


def write_checkpoint(path: Path, observations: int = 13) -> None:
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
        b"visual-13-v2" if observations == 13 else b"visual-matchup-95-v2",
        b"fixed-24-compatibility",
        0,
        18,
    )
    integrity = struct.pack("<QQ", len(payload), _fnv1a(payload))
    path.write_bytes(header + contract + integrity + payload)


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

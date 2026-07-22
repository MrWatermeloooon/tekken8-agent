from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from t8_agent.live.v2_policy import LiveV2GpuAgent, V2Checkpoint, _fnv1a
from t8_agent.sim.action_space import ACTION_SPACE
from t8_agent.vision.temporal import VisualEstimate


def write_checkpoint(path: Path) -> None:
    observations = 13
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
    header = struct.pack("<8sIIIIQQ", b"T8V2PPO\0", 2, observations, actions, hidden, 17, model_count)
    integrity = struct.pack("<QQ", len(payload), _fnv1a(payload))
    path.write_bytes(header + integrity + payload)


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


def test_native_checkpoint_rejects_corruption(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "corrupt.t8ppo"
    write_checkpoint(checkpoint_path)
    raw = bytearray(checkpoint_path.read_bytes())
    raw[-1] ^= 0x40
    checkpoint_path.write_bytes(raw)
    with pytest.raises(ValueError, match="integrity"):
        V2Checkpoint.load(checkpoint_path)

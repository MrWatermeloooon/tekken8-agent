from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from t8_agent.sim.action_space import action_count, index_to_action
from t8_agent.sim.actions import SimAction
from t8_agent.vision.temporal import VisualEstimate


_HEADER = struct.Struct("<8sIIIIQQ")
_INTEGRITY = struct.Struct("<QQ")
_MAGIC = b"T8V2PPO\0"
_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211


def _fnv1a(payload: bytes) -> int:
    checksum = _FNV_OFFSET
    for value in payload:
        checksum ^= value
        checksum = (checksum * _FNV_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return checksum


@dataclass(frozen=True)
class V2Checkpoint:
    observation_size: int
    action_count: int
    hidden_size: int
    optimizer_step: int
    weights_1: np.ndarray
    bias_1: np.ndarray
    weights_2: np.ndarray
    bias_2: np.ndarray
    weights_out: np.ndarray
    bias_out: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "V2Checkpoint":
        checkpoint_path = Path(path)
        raw = checkpoint_path.read_bytes()
        if len(raw) < _HEADER.size + _INTEGRITY.size:
            raise ValueError(f"truncated V2 checkpoint: {checkpoint_path}")
        magic, version, observations, actions, hidden, optimizer_step, parameter_count = (
            _HEADER.unpack_from(raw)
        )
        if magic != _MAGIC or version != 2:
            raise ValueError(f"live inference requires an integrity-protected V2 checkpoint: {checkpoint_path}")
        payload_bytes, expected_checksum = _INTEGRITY.unpack_from(raw, _HEADER.size)
        payload = raw[_HEADER.size + _INTEGRITY.size :]
        if len(payload) != payload_bytes or _fnv1a(payload) != expected_checksum:
            raise ValueError(f"V2 checkpoint integrity check failed: {checkpoint_path}")

        output_size = actions + 1
        w1_count = hidden * observations
        w2_count = hidden * hidden
        out_count = output_size * hidden
        expected_parameters = w1_count + hidden + w2_count + hidden + out_count + output_size
        if parameter_count != expected_parameters or payload_bytes != expected_parameters * 3 * 4:
            raise ValueError(f"V2 checkpoint architecture/payload mismatch: {checkpoint_path}")
        values = np.frombuffer(payload, dtype="<f4")
        offset = 0

        def take(count: int) -> np.ndarray:
            nonlocal offset
            result = values[offset : offset + count].copy()
            offset += count
            return result

        weights_1 = take(w1_count).reshape(hidden, observations)
        bias_1 = take(hidden)
        weights_2 = take(w2_count).reshape(hidden, hidden)
        bias_2 = take(hidden)
        weights_out = take(out_count).reshape(output_size, hidden)
        bias_out = take(output_size)
        return cls(
            observations,
            actions,
            hidden,
            optimizer_step,
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            weights_out,
            bias_out,
        )


class LiveV2GpuAgent:
    """Runs a native V2 13-feature checkpoint with Torch CUDA inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        deterministic: bool = True,
        player: int = 1,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('install the GPU live dependencies with: pip install -e ".[live]"') from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for V2 live inference but Torch cannot access a CUDA GPU")
        loaded = V2Checkpoint.load(checkpoint)
        if player not in (1, 2):
            raise ValueError(f"player must be 1 or 2, got {player}")
        if loaded.observation_size != 13 or loaded.action_count != action_count():
            raise ValueError(
                "live V2 checkpoint must have 13 observations and "
                f"{action_count()} actions; got {loaded.observation_size}/{loaded.action_count}"
            )
        self._torch = torch
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.player = player
        self.weights_1 = torch.as_tensor(loaded.weights_1, device=self.device)
        self.bias_1 = torch.as_tensor(loaded.bias_1, device=self.device)
        self.weights_2 = torch.as_tensor(loaded.weights_2, device=self.device)
        self.bias_2 = torch.as_tensor(loaded.bias_2, device=self.device)
        self.weights_out = torch.as_tensor(loaded.weights_out, device=self.device)
        self.bias_out = torch.as_tensor(loaded.bias_out, device=self.device)

    def _logits_tensor(self, observation: np.ndarray):
        torch = self._torch
        vector = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if tuple(vector.shape) != (13,):
            raise ValueError(f"V2 live observation must have shape (13,), got {tuple(vector.shape)}")
        with torch.inference_mode():
            hidden_1 = torch.tanh(torch.mv(self.weights_1, vector) + self.bias_1)
            hidden_2 = torch.tanh(torch.mv(self.weights_2, hidden_1) + self.bias_2)
            output = torch.mv(self.weights_out, hidden_2) + self.bias_out
        return output[: action_count()]

    def logits(self, observation: np.ndarray) -> np.ndarray:
        return self._logits_tensor(observation).cpu().numpy()

    def act(self, estimate: VisualEstimate) -> SimAction:
        torch = self._torch
        logits = self._logits_tensor(self.observation(estimate))
        if self.deterministic:
            index = int(torch.argmax(logits).item())
        else:
            index = int(torch.multinomial(torch.softmax(logits, dim=0), 1).item())
        return index_to_action(index)

    def observation(self, estimate: VisualEstimate) -> np.ndarray:
        vector = estimate.to_vector()
        if self.player == 1:
            return vector
        # The simulator's visual contract is player-relative. Swap every
        # own/opponent pair while leaving distance unchanged for a P2 agent.
        return vector[[1, 0, 3, 2, 4, 6, 5, 8, 7, 10, 9, 12, 11]]

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from t8_agent.roster.catalog import load_catalog
from t8_agent.roster.temporal import MatchupObservationEncoder, ScreenTemporalFrame, TemporalFrame
from t8_agent.sim.action_space import action_count, index_to_action
from t8_agent.sim.actions import SimAction
from t8_agent.vision.temporal import VisualEstimate


_HEADER = struct.Struct("<8sIIIIQQ")
_CONTRACT = struct.Struct("<64s24s32s32sII")
# Version 4+: enabled, reserved, sample count, clip, epsilon. The running
# mean and variance follow the optimizer tensors inside the payload.
_NORMALIZER = struct.Struct("<IIQff")
_INTEGRITY = struct.Struct("<QQ")
_MAGIC = b"T8V2PPO\0"
_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
SCREEN_CONTRACT = "screen-matchup-95-v1"


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
    catalog_sha256: str
    roster_version: str
    observation_contract: str
    action_contract: str
    action_feature_size: int
    universal_action_count: int
    weights_1: np.ndarray
    bias_1: np.ndarray
    weights_2: np.ndarray
    bias_2: np.ndarray
    weights_out: np.ndarray
    bias_out: np.ndarray
    # None when the checkpoint has no active normalizer (V3, disabled, or no
    # samples merged yet); inputs then pass through raw, as in training.
    observation_mean: np.ndarray | None = None
    observation_variance: np.ndarray | None = None
    observation_clip: float = 10.0
    observation_epsilon: float = 1e-8

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        if self.observation_mean is None or self.observation_variance is None:
            return observation
        scaled = (observation - self.observation_mean) / np.sqrt(
            self.observation_variance + np.float32(self.observation_epsilon))
        return np.clip(scaled, -self.observation_clip, self.observation_clip).astype(np.float32)

    @classmethod
    def load(cls, path: str | Path) -> "V2Checkpoint":
        checkpoint_path = Path(path)
        raw = checkpoint_path.read_bytes()
        if len(raw) < _HEADER.size + _CONTRACT.size + _INTEGRITY.size:
            raise ValueError(f"truncated V2 checkpoint: {checkpoint_path}")
        magic, version, observations, actions, hidden, optimizer_step, parameter_count = (
            _HEADER.unpack_from(raw)
        )
        if magic == _MAGIC and version < 3:
            raise ValueError(
                f"legacy fixed-action checkpoint is incompatible with the full-roster contract: "
                f"{checkpoint_path}"
            )
        if magic != _MAGIC or version not in (3, 4):
            raise ValueError(f"live inference requires a contract-bound V3/V4 checkpoint: {checkpoint_path}")
        contract = _CONTRACT.unpack_from(raw, _HEADER.size)
        decode = lambda value: value.split(b"\0", 1)[0].decode("ascii")
        catalog_sha256, roster_version, observation_contract, action_contract = map(decode, contract[:4])
        action_feature_size, universal_action_count = contract[4:]
        integrity_offset = _HEADER.size + _CONTRACT.size
        normalizer_enabled, normalizer_count = 0, 0
        observation_clip, observation_epsilon = 10.0, 1e-8
        if version >= 4:
            if len(raw) < integrity_offset + _NORMALIZER.size + _INTEGRITY.size:
                raise ValueError(f"truncated V2 checkpoint: {checkpoint_path}")
            normalizer_enabled, reserved, normalizer_count, observation_clip, observation_epsilon = (
                _NORMALIZER.unpack_from(raw, integrity_offset)
            )
            if (normalizer_enabled not in (0, 1) or reserved != 0 or
                    not (np.isfinite(observation_clip) and observation_clip > 0) or
                    not (np.isfinite(observation_epsilon) and observation_epsilon > 0)):
                raise ValueError(f"V2 checkpoint observation normalizer is invalid: {checkpoint_path}")
            integrity_offset += _NORMALIZER.size
        payload_bytes, expected_checksum = _INTEGRITY.unpack_from(raw, integrity_offset)
        payload = raw[integrity_offset + _INTEGRITY.size :]
        if len(payload) != payload_bytes or _fnv1a(payload) != expected_checksum:
            raise ValueError(f"V2 checkpoint integrity check failed: {checkpoint_path}")

        output_size = (action_feature_size if action_feature_size > 0 else actions) + 1
        w1_count = hidden * observations
        w2_count = hidden * hidden
        out_count = output_size * hidden
        expected_parameters = w1_count + hidden + w2_count + hidden + out_count + output_size
        normalizer_floats = 2 * observations if version >= 4 else 0
        if (parameter_count != expected_parameters or
                payload_bytes != (expected_parameters * 3 + normalizer_floats) * 4):
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
        observation_mean = observation_variance = None
        if version >= 4:
            offset = expected_parameters * 3  # skip Adam moments
            mean, variance = take(observations), take(observations)
            if not (np.isfinite(mean).all() and np.isfinite(variance).all() and (variance >= 0).all()):
                raise ValueError(f"V2 checkpoint observation statistics are invalid: {checkpoint_path}")
            if normalizer_enabled and normalizer_count > 0:
                observation_mean, observation_variance = mean, variance
        return cls(
            observations,
            actions,
            hidden,
            optimizer_step,
            catalog_sha256,
            roster_version,
            observation_contract,
            action_contract,
            action_feature_size,
            universal_action_count,
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            weights_out,
            bias_out,
            observation_mean,
            observation_variance,
            float(observation_clip),
            float(observation_epsilon),
        )


class LiveV2GpuAgent:
    """Runs native V2 visual or visual-matchup checkpoints with Torch inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        deterministic: bool = True,
        player: int = 1,
        opponent_character: str | int | None = None,
        opponent_archetype: str | int | None = None,
        data_root: str | Path | None = None,
        screen_position_sigma: float | None = None,
        screen_event_error: float | None = None,
        motion_threshold: float = 0.015,
        attack_cue_threshold: float = 0.5,
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
        if loaded.observation_size not in (13, 95) or loaded.action_count != action_count():
            raise ValueError(
                "live V2 checkpoint must have 13 visual or 95 visual-matchup observations and "
                f"{action_count()} actions; got {loaded.observation_size}/{loaded.action_count}"
            )
        self._torch = torch
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.player = player
        self.observation_size = loaded.observation_size
        self._matchup_encoder: MatchupObservationEncoder | None = None
        self._opponent_character_id: int | None = None
        self._opponent_archetype_id: int | None = None
        self._previous_estimated_move = 0
        self._repeated_move_frames = 0
        self._previous_own_health: float | None = None
        self._previous_opponent_health: float | None = None
        # Screen-only checkpoints were trained on binary activity/attack-cue
        # detections plus the measurement-uncertainty inputs, which must come
        # from calibration of this capture setup rather than defaults.
        self.screen_only = loaded.observation_contract == SCREEN_CONTRACT
        self._opponent_activity_frames = 0
        if self.screen_only:
            if screen_position_sigma is None or screen_event_error is None:
                raise ValueError(
                    "a screen-only checkpoint needs calibrated screen_position_sigma and screen_event_error; "
                    "run scripts/calibrate_screen_uncertainty.py"
                )
            if not (np.isfinite(screen_position_sigma) and screen_position_sigma >= 0 and
                    np.isfinite(screen_event_error) and 0 <= screen_event_error <= 0.5):
                raise ValueError("screen uncertainty must be finite, sigma >= 0 and event error in [0, 0.5]")
        self.screen_position_sigma = float(screen_position_sigma or 0.0)
        self.screen_event_error = float(screen_event_error or 0.0)
        self.motion_threshold = float(motion_threshold)
        self.attack_cue_threshold = float(attack_cue_threshold)
        if loaded.observation_size == 95:
            if opponent_character is None or opponent_archetype is None:
                raise ValueError(
                    "a 95-feature matchup checkpoint requires opponent_character and "
                    "opponent_archetype"
                )
            catalog = load_catalog(data_root)
            self._opponent_character_id = catalog.character(opponent_character).id
            self._opponent_archetype_id = _resolve_archetype_id(catalog.archetypes, opponent_archetype)
            self._matchup_encoder = MatchupObservationEncoder(base_size=13)
        self.weights_1 = torch.as_tensor(loaded.weights_1, device=self.device)
        self.bias_1 = torch.as_tensor(loaded.bias_1, device=self.device)
        self.weights_2 = torch.as_tensor(loaded.weights_2, device=self.device)
        self.bias_2 = torch.as_tensor(loaded.bias_2, device=self.device)
        self.weights_out = torch.as_tensor(loaded.weights_out, device=self.device)
        self.bias_out = torch.as_tensor(loaded.bias_out, device=self.device)
        self._checkpoint = loaded

    def _logits_tensor(self, observation: np.ndarray):
        torch = self._torch
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"V2 live observation must have shape ({self.observation_size},), "
                f"got {observation.shape}"
            )
        if not np.isfinite(observation).all():
            raise ValueError("V2 live observation contains non-finite values")
        vector = torch.as_tensor(self._checkpoint.normalize(observation), device=self.device)
        with torch.inference_mode():
            hidden_1 = torch.tanh(torch.mv(self.weights_1, vector) + self.bias_1)
            hidden_2 = torch.tanh(torch.mv(self.weights_2, hidden_1) + self.bias_2)
            output = torch.mv(self.weights_out, hidden_2) + self.bias_out
        return output[: action_count()]

    def logits(self, observation: np.ndarray) -> np.ndarray:
        return self._logits_tensor(observation).cpu().numpy()

    def act(self, estimate: VisualEstimate, *, temporal_frame: TemporalFrame | None = None,
            action_mask: np.ndarray | None = None) -> SimAction:
        torch = self._torch
        logits = self._logits_tensor(self.observation(estimate, temporal_frame=temporal_frame))
        if not torch.isfinite(logits).all():
            raise ValueError("policy produced invalid logits")
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            if mask.shape != (action_count(),) or not mask.any():
                raise ValueError("action mask must have one entry per action and a legal action")
            logits = logits.masked_fill(~torch.as_tensor(mask, device=self.device), -torch.inf)
        if self.deterministic:
            index = int(torch.argmax(logits).item())
        else:
            index = int(torch.multinomial(torch.softmax(logits, dim=0), 1).item())
        return index_to_action(index)

    def observation(
        self,
        estimate: VisualEstimate,
        *,
        temporal_frame: TemporalFrame | None = None,
    ) -> np.ndarray:
        vector = estimate.to_vector()
        if self.screen_only:
            # Same binary detections the simulator's screen contract produces.
            vector[7] = float(estimate.p1_motion >= self.motion_threshold)
            vector[8] = float(estimate.p2_motion >= self.motion_threshold)
            vector[11] = float(estimate.p1_attack_likelihood >= self.attack_cue_threshold)
            vector[12] = float(estimate.p2_attack_likelihood >= self.attack_cue_threshold)
        if self.player == 2:
            # The simulator's visual contract is player-relative. Swap every
            # own/opponent pair while leaving distance unchanged for a P2 agent.
            vector = vector[[1, 0, 3, 2, 4, 6, 5, 8, 7, 10, 9, 12, 11]]
        if self._matchup_encoder is None:
            return vector
        frame = (self._screen_temporal_frame(vector) if self.screen_only
                 else temporal_frame or self._estimated_temporal_frame(estimate))
        return self._matchup_encoder.encode(
            vector,
            opponent_character_id=self._opponent_character_id,
            opponent_archetype_id=self._opponent_archetype_id,
            frame=frame,
        )

    def reset_episode(self) -> None:
        """Clears the temporal stack between rounds or opponents."""
        if self._matchup_encoder is not None:
            self._matchup_encoder.reset()
        self._previous_estimated_move = 0
        self._repeated_move_frames = 0
        self._previous_own_health = None
        self._previous_opponent_health = None
        self._opponent_activity_frames = 0

    def _screen_temporal_frame(self, vector: np.ndarray) -> ScreenTemporalFrame:
        """Screen-contract history step from the player-relative base vector."""
        own_health, opponent_health = float(vector[0]), float(vector[1])
        outcome = 0.0
        if self._previous_own_health is not None and self._previous_opponent_health is not None:
            outcome = float(np.clip(
                ((self._previous_opponent_health - opponent_health) -
                 (self._previous_own_health - own_health)) * 10.0,
                -1.0,
                1.0,
            ))
        self._previous_own_health = own_health
        self._previous_opponent_health = opponent_health
        opponent_active = bool(vector[8] > 0.5)
        self._opponent_activity_frames = min(60, self._opponent_activity_frames + 4) if opponent_active else 0
        return ScreenTemporalFrame(
            opponent_activity=opponent_active,
            opponent_attack_cue=bool(vector[12] > 0.5),
            position_sigma=self.screen_position_sigma,
            event_error=self.screen_event_error,
            activity_frames=self._opponent_activity_frames,
            outcome=outcome,
            distance=float(vector[4]),
            side_movement=float(np.clip(vector[6], -1.0, 1.0)),
        )

    def _estimated_temporal_frame(self, estimate: VisualEstimate) -> TemporalFrame:
        if self.player == 1:
            opponent_velocity = estimate.p2_velocity
            own_health = estimate.p1_health_ratio
            opponent_health = estimate.p2_health_ratio
        else:
            opponent_velocity = estimate.p1_velocity
            own_health = estimate.p2_health_ratio
            opponent_health = estimate.p1_health_ratio
        # Motion alone cannot identify a move, its hit level, or recovery phase.
        # Negative values explicitly mean unknown, rather than inventing a jab.
        outcome = 0.0
        if self._previous_own_health is not None and self._previous_opponent_health is not None:
            outcome = float(np.clip(
                ((self._previous_opponent_health - opponent_health) -
                 (self._previous_own_health - own_health)) * 10.0,
                -1.0,
                1.0,
            ))
        self._previous_own_health = own_health
        self._previous_opponent_health = opponent_health
        return TemporalFrame(
            move_id=-1,
            animation_phase=-1.0,
            stance_id=-1,
            hit_level=-1,
            delay_frames=-1,
            outcome=outcome,
            distance=estimate.distance,
            side_movement=float(np.clip(opponent_velocity, -1.0, 1.0)),
        )


def _resolve_archetype_id(archetypes, value: str | int) -> int:
    for archetype in archetypes:
        if archetype.id == value or archetype.key == value:
            return archetype.id
    raise ValueError(f"unknown opponent archetype: {value}")

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

from t8_agent.sim.action_space import index_to_action, legal_action_mask
from t8_agent.sim.opponents import SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import SimAction, TekkenLiteEnv
from t8_agent.sim.observations import vector_observation
from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv

OpponentPolicy = Callable[[TekkenLiteEnv, int], SimAction]


def vecnormalize_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint).with_suffix(".vecnormalize.pkl")


class PpoCheckpointOpponent:
    def __init__(self, checkpoint: str | Path, deterministic: bool = True) -> None:
        from sb3_contrib import MaskablePPO

        self.checkpoint = Path(checkpoint)
        self.model = MaskablePPO.load(self.checkpoint)
        self.normalizer = None
        normalizer_path = vecnormalize_path(self.checkpoint)
        if normalizer_path.exists():
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            dummy_env = DummyVecEnv([lambda: TekkenLiteSingleAgentEnv()])
            self.normalizer = VecNormalize.load(str(normalizer_path), dummy_env)
            self.normalizer.training = False
            self.normalizer.norm_reward = False
        self.deterministic = deterministic

    def __call__(self, env: TekkenLiteEnv, player: int) -> SimAction:
        obs = vector_observation(env.state, env.config, player)
        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
        mask = legal_action_mask(env.state, player)
        action, _state = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        return index_to_action(int(action))


class OpponentPool:
    def __init__(
        self,
        scripted_names: Sequence[str],
        checkpoint_paths: Sequence[str | Path] | None = None,
        checkpoint_ratings: dict[str, float] | None = None,
        target_rating: float | None = None,
        scripted_sample_rate: float = 0.35,
        old_checkpoint_sample_rate: float = 0.15,
        max_recent_checkpoints: int = 8,
        rng: random.Random | None = None,
    ) -> None:
        self.scripted_names = list(scripted_names)
        if not 0.0 <= scripted_sample_rate <= 1.0:
            raise ValueError("scripted_sample_rate must be between 0 and 1")
        if not 0.0 <= old_checkpoint_sample_rate <= 1.0:
            raise ValueError("old_checkpoint_sample_rate must be between 0 and 1")
        if max_recent_checkpoints < 1:
            raise ValueError("max_recent_checkpoints must be at least 1")
        self.scripted_sample_rate = scripted_sample_rate
        self.old_checkpoint_sample_rate = old_checkpoint_sample_rate
        self.max_recent_checkpoints = max_recent_checkpoints
        self.checkpoint_ratings = checkpoint_ratings or {}
        self.target_rating = target_rating
        self.rng = rng or random.Random()
        unknown = [name for name in self.scripted_names if name not in SCRIPTED_POLICIES]
        if unknown:
            known = ", ".join(sorted(SCRIPTED_POLICIES))
            raise ValueError(f"unknown opponent(s) {unknown}; known: {known}")
        self.scripted: list[tuple[str, OpponentPolicy]] = [
            (name, SCRIPTED_POLICIES[name]) for name in self.scripted_names
        ]
        self.checkpoints = [Path(path) for path in checkpoint_paths or []]
        self._loaded: dict[Path, PpoCheckpointOpponent] = {}

    def sample(self) -> tuple[str, OpponentPolicy]:
        if not self.checkpoints or self.rng.random() < self.scripted_sample_rate:
            return self.rng.choice(self.scripted)

        if self.checkpoint_ratings and self.target_rating is not None:
            path = self._sample_by_rating()
            return f"checkpoint:{path.name}", self._load(path)

        recent = self.checkpoints[-self.max_recent_checkpoints :]
        older = self.checkpoints[: -self.max_recent_checkpoints]
        if older and self.rng.random() < self.old_checkpoint_sample_rate:
            path = self.rng.choice(older)
        else:
            path = self.rng.choice(recent)
        return f"checkpoint:{path.name}", self._load(path)

    def _sample_by_rating(self) -> Path:
        weighted: list[tuple[Path, float]] = []
        for path in self.checkpoints:
            rating = self.checkpoint_ratings.get(str(path), self.target_rating or 1000.0)
            distance = abs(rating - (self.target_rating or rating))
            weighted.append((path, 1.0 / (1.0 + distance / 100.0)))
        total = sum(weight for _path, weight in weighted)
        roll = self.rng.random() * total
        upto = 0.0
        for path, weight in weighted:
            upto += weight
            if upto >= roll:
                return path
        return weighted[-1][0]

    def _load(self, path: Path) -> PpoCheckpointOpponent:
        if path not in self._loaded:
            self._loaded[path] = PpoCheckpointOpponent(path)
        return self._loaded[path]

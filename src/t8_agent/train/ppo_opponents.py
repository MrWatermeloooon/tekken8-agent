from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

from t8_agent.sim.action_space import index_to_action, legal_action_mask
from t8_agent.sim.opponents import SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import SimAction, TekkenLiteEnv
from t8_agent.sim.observations import vector_observation

OpponentPolicy = Callable[[TekkenLiteEnv, int], SimAction]


class PpoCheckpointOpponent:
    def __init__(self, checkpoint: str | Path, deterministic: bool = True) -> None:
        from sb3_contrib import MaskablePPO

        self.checkpoint = Path(checkpoint)
        self.model = MaskablePPO.load(self.checkpoint)
        self.deterministic = deterministic

    def __call__(self, env: TekkenLiteEnv, player: int) -> SimAction:
        obs = vector_observation(env.state, env.config, player)
        mask = legal_action_mask(env.state, player)
        action, _state = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        return index_to_action(int(action))


class OpponentPool:
    def __init__(
        self,
        scripted_names: Sequence[str],
        checkpoint_paths: Sequence[str | Path] | None = None,
        old_checkpoint_sample_rate: float = 0.15,
        max_recent_checkpoints: int = 8,
        rng: random.Random | None = None,
    ) -> None:
        self.scripted_names = list(scripted_names)
        self.old_checkpoint_sample_rate = old_checkpoint_sample_rate
        self.max_recent_checkpoints = max_recent_checkpoints
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
        if not self.checkpoints or self.rng.random() < 0.35:
            return self.rng.choice(self.scripted)

        recent = self.checkpoints[-self.max_recent_checkpoints :]
        older = self.checkpoints[: -self.max_recent_checkpoints]
        if older and self.rng.random() < self.old_checkpoint_sample_rate:
            path = self.rng.choice(older)
        else:
            path = self.rng.choice(recent)
        return f"checkpoint:{path.name}", self._load(path)

    def _load(self, path: Path) -> PpoCheckpointOpponent:
        if path not in self._loaded:
            self._loaded[path] = PpoCheckpointOpponent(path)
        return self._loaded[path]

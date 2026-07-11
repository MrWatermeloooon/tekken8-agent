from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from t8_agent.core.types import GameState
from t8_agent.sim.action_space import ACTION_SPACE, index_to_action, legal_action_mask
from t8_agent.sim.observations import vector_observation
from t8_agent.sim.opponents import SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import FighterRuntime, SimAction, SimConfig, SimState, TekkenLiteEnv
from t8_agent.train.ppo_opponents import vecnormalize_path


LEGACY_ACTION_SPACE_NO_THROW_BREAKS = [
    SimAction.NEUTRAL,
    SimAction.WALK_FORWARD,
    SimAction.WALK_BACK,
    SimAction.DASH_FORWARD,
    SimAction.DASH_BACK,
    SimAction.CROUCH,
    SimAction.STAND,
    SimAction.JUMP,
    SimAction.SIDESTEP_LEFT,
    SimAction.SIDESTEP_RIGHT,
    SimAction.SIDEWALK_LEFT,
    SimAction.SIDEWALK_RIGHT,
    SimAction.BLOCK_HIGH,
    SimAction.BLOCK_LOW,
    SimAction.LOW_PARRY,
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]


class LiveScriptedAgent:
    """Small live-test policy for validating screen capture and controller output."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.tick = 0

    def act(self, state: GameState) -> SimAction:
        self.tick += 1
        if state.round_over or state.p1.health <= 0:
            return SimAction.NEUTRAL
        if self.tick % 7 == 0:
            return self.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW, SimAction.WALK_BACK])
        if self.tick % 3 == 0:
            return SimAction.WALK_FORWARD
        return self.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.F2])


class LivePpoCheckpointAgent:
    """Runs a simulator PPO checkpoint against a coarse live-screen state."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config: SimConfig | None = None,
        deterministic: bool = True,
    ) -> None:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError(
                "sb3-contrib is not installed. Install RL extras with: "
                '.\\.venv\\Scripts\\python -m pip install -e ".[rl]"'
            ) from exc
        self.checkpoint = Path(checkpoint)
        self.model = MaskablePPO.load(self.checkpoint, device="cpu")
        self.observation_dim = int(self.model.observation_space.shape[0])
        self.action_dim = int(self.model.action_space.n)
        if self.observation_dim not in {18, 19}:
            raise RuntimeError(f"unsupported checkpoint observation size: {self.observation_dim}")
        if self.action_dim not in {len(LEGACY_ACTION_SPACE_NO_THROW_BREAKS), len(ACTION_SPACE)}:
            raise RuntimeError(f"unsupported checkpoint action count: {self.action_dim}")
        self.normalizer = self._load_normalizer()
        self.config = config or SimConfig()
        self.deterministic = deterministic
        self.shadow_env = TekkenLiteEnv(config=self.config, seed=9090)
        self.opponent_policy = SCRIPTED_POLICIES["rushdown"]

    def act(self, state: GameState) -> SimAction:
        sim_state = self._sync_shadow_health(state)
        obs = self._observation(sim_state)
        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
        mask = self._action_mask(sim_state)
        action, _model_state = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        sim_action = self._index_to_action(int(action))
        opponent_action = self.opponent_policy(self.shadow_env, 2)
        result = self.shadow_env.step(sim_action, opponent_action)
        if result.terminated or result.truncated:
            self.shadow_env.reset()
        return sim_action

    @property
    def compatibility_label(self) -> str:
        return f"obs={self.observation_dim} actions={self.action_dim}"

    def _load_normalizer(self):
        normalizer_path = vecnormalize_path(self.checkpoint)
        if not normalizer_path.exists():
            return None
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        dummy_env = DummyVecEnv([lambda: _make_observation_shape_env(self.observation_dim, self.action_dim)])
        normalizer = VecNormalize.load(str(normalizer_path), dummy_env)
        normalizer.training = False
        normalizer.norm_reward = False
        return normalizer

    def _observation(self, state: SimState) -> np.ndarray:
        obs = vector_observation(state, self.shadow_env.config, player=1)
        if self.observation_dim == obs.shape[0]:
            return obs
        if self.observation_dim == 18 and obs.shape[0] == 19:
            return np.concatenate([obs[:17], obs[18:19]]).astype(np.float32)
        raise RuntimeError(f"cannot adapt observation shape {obs.shape[0]} to checkpoint shape {self.observation_dim}")

    def _action_mask(self, state: SimState) -> np.ndarray:
        mask = legal_action_mask(state, player=1)
        if self.action_dim == len(mask):
            return mask
        if self.action_dim == len(LEGACY_ACTION_SPACE_NO_THROW_BREAKS):
            return np.array([mask[ACTION_SPACE.index(action)] for action in LEGACY_ACTION_SPACE_NO_THROW_BREAKS], dtype=bool)
        raise RuntimeError(f"cannot adapt action mask length {len(mask)} to checkpoint action count {self.action_dim}")

    def _index_to_action(self, index: int) -> SimAction:
        if self.action_dim == len(ACTION_SPACE):
            return index_to_action(index)
        return LEGACY_ACTION_SPACE_NO_THROW_BREAKS[index]

    def _sync_shadow_health(self, state: GameState) -> SimState:
        sim_state = self.shadow_env.state
        p1_x = float((state.raw or {}).get("p1_x", state.p1.position_x))
        p2_x = float((state.raw or {}).get("p2_x", state.p2.position_x))
        sim_state = SimState(
            p1=FighterRuntime(
                health=float(state.p1.health),
                x=p1_x,
                y=float(state.p1.position_y),
                guard=sim_state.p1.guard,
                move_key=sim_state.p1.move_key,
                move_frame=sim_state.p1.move_frame,
                has_hit=sim_state.p1.has_hit,
                hitstun=sim_state.p1.hitstun,
                blockstun=sim_state.p1.blockstun,
                airborne=sim_state.p1.airborne,
                throw_break_active=sim_state.p1.throw_break_active,
                launches_taken=sim_state.p1.launches_taken,
                whiffs=sim_state.p1.whiffs,
            ),
            p2=FighterRuntime(
                health=float(state.p2.health),
                x=p2_x,
                y=float(state.p2.position_y),
                guard=sim_state.p2.guard,
                move_key=sim_state.p2.move_key,
                move_frame=sim_state.p2.move_frame,
                has_hit=sim_state.p2.has_hit,
                hitstun=sim_state.p2.hitstun,
                blockstun=sim_state.p2.blockstun,
                airborne=sim_state.p2.airborne,
                throw_break_active=sim_state.p2.throw_break_active,
                launches_taken=sim_state.p2.launches_taken,
                whiffs=sim_state.p2.whiffs,
            ),
            frame=sim_state.frame,
            round_over=state.round_over,
            winner=state.winner,
        )
        self.shadow_env.state = sim_state
        return sim_state


def find_latest_checkpoint(root: str | Path = "checkpoints") -> Path:
    root = Path(root)
    candidates = sorted(root.rglob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no PPO checkpoints found under {root}")
    return candidates[0]


def _make_observation_shape_env(observation_dim: int, action_dim: int):
    import gymnasium as gym
    from gymnasium import spaces

    class ObservationShapeEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(action_dim)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            _ = options
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action: int):
            _ = action
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

    return ObservationShapeEnv()

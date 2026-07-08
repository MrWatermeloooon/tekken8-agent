from __future__ import annotations

import random
from pathlib import Path

from t8_agent.core.types import GameState
from t8_agent.sim.action_space import index_to_action, legal_action_mask
from t8_agent.sim.observations import vector_observation
from t8_agent.sim.opponents import SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import FighterRuntime, SimAction, SimConfig, SimState, TekkenLiteEnv


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
        self.model = MaskablePPO.load(self.checkpoint)
        self.config = config or SimConfig()
        self.deterministic = deterministic
        self.shadow_env = TekkenLiteEnv(config=self.config, seed=9090)
        self.opponent_policy = SCRIPTED_POLICIES["rushdown"]

    def act(self, state: GameState) -> SimAction:
        sim_state = self._sync_shadow_health(state)
        obs = vector_observation(sim_state, self.shadow_env.config, player=1)
        mask = legal_action_mask(sim_state, player=1)
        action, _model_state = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        sim_action = index_to_action(int(action))
        opponent_action = self.opponent_policy(self.shadow_env, 2)
        result = self.shadow_env.step(sim_action, opponent_action)
        if result.terminated or result.truncated:
            self.shadow_env.reset()
        return sim_action

    def _sync_shadow_health(self, state: GameState) -> SimState:
        sim_state = self.shadow_env.state
        p1_x = float((state.raw or {}).get("p1_x", -0.65))
        p2_x = float((state.raw or {}).get("p2_x", 0.65))
        sim_state = SimState(
            p1=FighterRuntime(
                health=float(state.p1.health),
                x=p1_x,
                guard=sim_state.p1.guard,
                move_key=sim_state.p1.move_key,
                move_frame=sim_state.p1.move_frame,
                has_hit=sim_state.p1.has_hit,
                hitstun=sim_state.p1.hitstun,
                blockstun=sim_state.p1.blockstun,
            ),
            p2=FighterRuntime(
                health=float(state.p2.health),
                x=p2_x,
                guard=sim_state.p2.guard,
                move_key=sim_state.p2.move_key,
                move_frame=sim_state.p2.move_frame,
                has_hit=sim_state.p2.has_hit,
                hitstun=sim_state.p2.hitstun,
                blockstun=sim_state.p2.blockstun,
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

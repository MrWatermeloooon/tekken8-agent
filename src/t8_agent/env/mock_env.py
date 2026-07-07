from __future__ import annotations

import random
from dataclasses import replace

from t8_agent.core.types import DiscreteAction, GameState, PlayerState, StepResult
from t8_agent.io.input_backend import InputBackend, MockInputBackend


class MockTekkenEnv:
    """Tiny deterministic-ish environment for testing loops without Tekken 8."""

    def __init__(self, input_backend: InputBackend | None = None, max_steps: int = 600) -> None:
        self.input_backend = input_backend or MockInputBackend()
        self.max_steps = max_steps
        self.steps = 0
        self.state = self._initial_state()

    def reset(self) -> GameState:
        self.steps = 0
        self.state = self._initial_state()
        self.input_backend.release_all()
        return self.state

    def step(self, action: DiscreteAction) -> StepResult:
        previous = self.state
        self.input_backend.send(action)
        self.steps += 1
        self.state = self._transition(previous, action)

        damage_dealt = previous.p2.health - self.state.p2.health
        damage_taken = previous.p1.health - self.state.p1.health
        reward = damage_dealt - damage_taken
        if self.state.winner == 1:
            reward += 50.0
        elif self.state.winner == 2:
            reward -= 50.0
        if action == DiscreteAction.NEUTRAL:
            reward -= 0.01

        truncated = self.steps >= self.max_steps and not self.state.round_over
        return StepResult(
            observation=self.state,
            reward=reward,
            terminated=self.state.round_over,
            truncated=truncated,
            info={
                "damage_dealt": damage_dealt,
                "damage_taken": damage_taken,
                "steps": self.steps,
            },
        )

    def close(self) -> None:
        self.input_backend.close()

    def _transition(self, state: GameState, action: DiscreteAction) -> GameState:
        p1_x = state.p1.position_x
        p2_x = state.p2.position_x
        p1_health = state.p1.health
        p2_health = state.p2.health

        if action == DiscreteAction.WALK_FORWARD:
            p1_x += 0.08
        elif action == DiscreteAction.WALK_BACK:
            p1_x -= 0.08
        elif action in {
            DiscreteAction.LEFT_PUNCH,
            DiscreteAction.RIGHT_PUNCH,
            DiscreteAction.LEFT_KICK,
            DiscreteAction.RIGHT_KICK,
        }:
            if abs(p1_x - p2_x) < 0.8:
                p2_health = max(0.0, p2_health - random.choice([3.0, 5.0, 8.0]))

        if random.random() < 0.04 and abs(p1_x - p2_x) < 0.8:
            p1_health = max(0.0, p1_health - random.choice([3.0, 5.0]))

        timer = max(0.0, state.round_timer - 0.1)
        winner = None
        if p2_health <= 0.0:
            winner = 1
        elif p1_health <= 0.0:
            winner = 2
        elif timer <= 0.0:
            winner = 1 if p1_health > p2_health else 2

        round_over = winner is not None
        return replace(
            state,
            p1=PlayerState(health=p1_health, position_x=p1_x),
            p2=PlayerState(health=p2_health, position_x=p2_x, facing=-1),
            round_timer=timer,
            round_over=round_over,
            winner=winner,
        )

    @staticmethod
    def _initial_state() -> GameState:
        return GameState(
            p1=PlayerState(health=180.0, position_x=-1.0),
            p2=PlayerState(health=180.0, position_x=1.0, facing=-1),
            round_timer=60.0,
        )


def main() -> None:
    env = MockTekkenEnv()
    observation = env.reset()
    total_reward = 0.0
    print(f"reset distance={observation.distance:.2f}")
    for _ in range(100):
        action = random.choice(list(DiscreteAction))
        result = env.step(action)
        total_reward += result.reward
        if result.terminated or result.truncated:
            break
    print(
        "episode "
        f"steps={env.steps} "
        f"p1_hp={env.state.p1.health:.1f} "
        f"p2_hp={env.state.p2.health:.1f} "
        f"winner={env.state.winner} "
        f"reward={total_reward:.2f}"
    )


if __name__ == "__main__":
    main()

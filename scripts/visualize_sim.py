from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, BOTTOM, Canvas, Label, Tk

from t8_agent.sim.action_space import index_to_action, legal_action_mask
from t8_agent.sim.moves import JUN_MOVES
from t8_agent.sim.observations import vector_observation
from t8_agent.sim.opponents import SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import SimAction, TekkenLiteEnv
from t8_agent.train.linear_policy import LinearPolicy
from t8_agent.train.ppo_opponents import vecnormalize_path
from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv


@dataclass
class PolicySpec:
    kind: str
    checkpoint: Path | None = None
    scripted_name: str = "poke"
    label: str = ""


class PolicyController:
    def __init__(self, spec: PolicySpec) -> None:
        self.spec = spec
        self.policy = None
        self.ppo_model = None
        self.normalizer = None
        if spec.kind == "checkpoint" and spec.checkpoint:
            if spec.checkpoint.suffix.lower() == ".zip":
                from sb3_contrib import MaskablePPO

                self.ppo_model = MaskablePPO.load(spec.checkpoint)
                normalizer_path = vecnormalize_path(spec.checkpoint)
                if normalizer_path.exists():
                    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                    dummy_env = DummyVecEnv([lambda: TekkenLiteSingleAgentEnv()])
                    self.normalizer = VecNormalize.load(str(normalizer_path), dummy_env)
                    self.normalizer.training = False
                    self.normalizer.norm_reward = False
            else:
                self.policy = LinearPolicy.load(spec.checkpoint)

    def act(self, env: TekkenLiteEnv, player: int) -> SimAction:
        own = env.state.p1 if player == 1 else env.state.p2
        if own.busy:
            return SimAction.NEUTRAL
        if self.spec.kind == "checkpoint":
            if self.ppo_model is not None:
                obs = vector_observation(env.state, env.config, player)
                if self.normalizer is not None:
                    obs = self.normalizer.normalize_obs(obs)
                mask = legal_action_mask(env.state, player)
                action, _state = self.ppo_model.predict(obs, deterministic=True, action_masks=mask)
                return index_to_action(int(action))
            if self.policy is None:
                raise RuntimeError("checkpoint policy requested without loaded checkpoint")
            return self.policy.act(env, player=player)
        if self.spec.kind == "random":
            return env.sample_action()
        return SCRIPTED_POLICIES[self.spec.scripted_name](env, player)

    @property
    def label(self) -> str:
        if self.spec.label:
            return self.spec.label
        if self.spec.kind == "checkpoint" and self.spec.checkpoint:
            return self.spec.checkpoint.stem
        if self.spec.kind == "random":
            return "random"
        return self.spec.scripted_name


class SimVisualizer:
    def __init__(
        self,
        env: TekkenLiteEnv,
        p1: PolicyController,
        p2: PolicyController,
        width: int = 1100,
        height: int = 620,
        steps_per_tick: int = 2,
    ) -> None:
        self.env = env
        self.p1 = p1
        self.p2 = p2
        self.width = width
        self.height = height
        self.steps_per_tick = steps_per_tick
        self.paused = False
        self.last_p1_action = SimAction.NEUTRAL
        self.last_p2_action = SimAction.NEUTRAL
        self.last_reward = 0.0
        self.episode = 1
        self.root = Tk()
        self.root.title("Tekken-lite Simulator Visualizer")
        self.canvas = Canvas(self.root, width=width, height=height, bg="#101418", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.status = Label(
            self.root,
            text="Space: pause | R: reset | +/-: speed | default: trained P1 vs active rushdown P2",
            anchor="w",
            bg="#101418",
            fg="#d8e1e8",
        )
        self.status.pack(side=BOTTOM, fill="x")
        self.root.bind("<space>", lambda _event: self.toggle_pause())
        self.root.bind("r", lambda _event: self.reset())
        self.root.bind("R", lambda _event: self.reset())
        self.root.bind("+", lambda _event: self.change_speed(1))
        self.root.bind("=", lambda _event: self.change_speed(1))
        self.root.bind("-", lambda _event: self.change_speed(-1))

    def run(self) -> None:
        self.tick()
        self.root.mainloop()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def reset(self) -> None:
        self.episode += 1
        self.env.reset()
        self.last_reward = 0.0
        self.last_p1_action = SimAction.NEUTRAL
        self.last_p2_action = SimAction.NEUTRAL

    def change_speed(self, delta: int) -> None:
        self.steps_per_tick = max(1, min(12, self.steps_per_tick + delta))

    def tick(self) -> None:
        if not self.paused:
            for _ in range(self.steps_per_tick):
                self.last_p1_action = self.p1.act(self.env, 1)
                self.last_p2_action = self.p2.act(self.env, 2)
                result = self.env.step(self.last_p1_action, self.last_p2_action)
                self.last_reward = result.reward_p1
                if result.terminated or result.truncated:
                    self.reset()
                    break
        self.draw()
        self.root.after(33, self.tick)

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        self._draw_background()
        self._draw_health_bars()
        self._draw_stage()
        self._draw_fighter(player=1)
        self._draw_fighter(player=2)
        self._draw_attack_range(player=1)
        self._draw_attack_range(player=2)
        self._draw_text()

    def _draw_background(self) -> None:
        c = self.canvas
        c.create_rectangle(0, 0, self.width, self.height, fill="#101418", outline="")
        c.create_rectangle(0, 0, self.width, 130, fill="#151d24", outline="")

    def _draw_health_bars(self) -> None:
        margin = 42
        bar_w = 430
        bar_h = 24
        y = 34
        self._health_bar(margin, y, bar_w, bar_h, self.env.state.p1.health, "#68d391", anchor_right=False)
        self._health_bar(self.width - margin - bar_w, y, bar_w, bar_h, self.env.state.p2.health, "#f687b3", anchor_right=True)
        timer = max(0, int((self.env.config.max_frames - self.env.state.frame) / 60))
        self.canvas.create_text(self.width / 2, y + 12, text=str(timer), fill="#f7fafc", font=("Segoe UI", 26, "bold"))

    def _health_bar(self, x: float, y: float, w: float, h: float, health: float, color: str, anchor_right: bool) -> None:
        ratio = max(0.0, min(1.0, health / self.env.config.max_health))
        self.canvas.create_rectangle(x, y, x + w, y + h, fill="#2d3748", outline="#607080")
        fill_w = w * ratio
        if anchor_right:
            self.canvas.create_rectangle(x + w - fill_w, y, x + w, y + h, fill=color, outline="")
        else:
            self.canvas.create_rectangle(x, y, x + fill_w, y + h, fill=color, outline="")

    def _draw_stage(self) -> None:
        y = self._floor_y()
        left = self._world_to_canvas_x(-self.env.config.stage_half_width)
        right = self._world_to_canvas_x(self.env.config.stage_half_width)
        self.canvas.create_line(left, y, right, y, fill="#7f8ea3", width=4)
        self.canvas.create_line(left, y - 155, left, y + 18, fill="#e2e8f0", width=3)
        self.canvas.create_line(right, y - 155, right, y + 18, fill="#e2e8f0", width=3)
        for idx in range(9):
            x = left + (right - left) * idx / 8
            self.canvas.create_line(x, y - 8, x, y + 8, fill="#4a5568", width=1)

    def _draw_fighter(self, player: int) -> None:
        fighter = self.env.state.p1 if player == 1 else self.env.state.p2
        color = "#68d391" if player == 1 else "#f687b3"
        outline = "#f6e05e" if fighter.hitstun > 0 else "#edf2f7"
        if fighter.blockstun > 0 or fighter.guard is not None:
            outline = "#63b3ed"
        x = self._world_to_canvas_x(fighter.x)
        floor = self._floor_y()
        w = 42
        h = 118
        self.canvas.create_rectangle(x - w / 2, floor - h, x + w / 2, floor, fill=color, outline=outline, width=3)
        head_r = 18
        self.canvas.create_oval(x - head_r, floor - h - 36, x + head_r, floor - h, fill=color, outline=outline, width=3)
        controller = self.p1 if player == 1 else self.p2
        label = f"P{player} {controller.label}"
        self.canvas.create_text(x, floor + 28, text=label, fill="#e2e8f0", font=("Segoe UI", 12, "bold"))

    def _draw_attack_range(self, player: int) -> None:
        fighter = self.env.state.p1 if player == 1 else self.env.state.p2
        if fighter.move_key is None:
            return
        move = JUN_MOVES[fighter.move_key]
        direction = 1 if player == 1 else -1
        start = self._world_to_canvas_x(fighter.x)
        end = self._world_to_canvas_x(fighter.x + move.range * direction)
        y = self._floor_y() - 82
        color = "#f6e05e" if fighter.move_frame > move.startup and fighter.move_frame <= move.startup + move.active else "#718096"
        self.canvas.create_line(start, y, end, y, fill=color, width=8)
        self.canvas.create_text((start + end) / 2, y - 20, text=move.command, fill=color, font=("Segoe UI", 11, "bold"))

    def _draw_text(self) -> None:
        state = self.env.state
        p1_move = state.p1.move_key or "-"
        p2_move = state.p2.move_key or "-"
        lines = [
            f"Episode {self.episode} | Frame {state.frame} | Speed {self.steps_per_tick}x | Last reward {self.last_reward:.2f}",
            f"P1 action: {self.last_p1_action.value} | move: {p1_move} | hitstun {state.p1.hitstun} | blockstun {state.p1.blockstun}",
            f"P2 action: {self.last_p2_action.value} | move: {p2_move} | hitstun {state.p2.hitstun} | blockstun {state.p2.blockstun}",
            f"Distance {state.distance:.2f} | P1 whiffs {state.p1.whiffs} | P2 whiffs {state.p2.whiffs}",
        ]
        for idx, line in enumerate(lines):
            self.canvas.create_text(42, 86 + idx * 22, text=line, anchor="w", fill="#d8e1e8", font=("Consolas", 12))
        if self.paused:
            self.canvas.create_text(self.width / 2, 190, text="PAUSED", fill="#f6e05e", font=("Segoe UI", 28, "bold"))

    def _world_to_canvas_x(self, x: float) -> float:
        left = 84
        right = self.width - 84
        normalized = (x + self.env.config.stage_half_width) / (self.env.config.stage_half_width * 2)
        return left + normalized * (right - left)

    def _floor_y(self) -> float:
        return self.height - 118


def build_policy(kind: str, checkpoint: str, scripted_name: str, label: str = "") -> PolicyController:
    if kind == "checkpoint":
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return PolicyController(PolicySpec(kind="checkpoint", checkpoint=path, label=label))
    if kind == "random":
        return PolicyController(PolicySpec(kind="random", label=label))
    if scripted_name not in SCRIPTED_POLICIES:
        known = ", ".join(sorted(SCRIPTED_POLICIES))
        raise ValueError(f"unknown scripted policy {scripted_name!r}; known: {known}")
    return PolicyController(PolicySpec(kind="scripted", scripted_name=scripted_name, label=label))


def run_headless(env: TekkenLiteEnv, p1: PolicyController, p2: PolicyController, steps: int) -> None:
    for _ in range(steps):
        result = env.step(p1.act(env, 1), p2.act(env, 2))
        if result.terminated or result.truncated:
            env.reset()
    print(
        f"headless_ok steps={steps} frame={env.state.frame} "
        f"p1_hp={env.state.p1.health:.1f} p2_hp={env.state.p2.health:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize the Tekken-lite simulator.")
    parser.add_argument("--p1", choices=["checkpoint", "scripted", "random"], default="checkpoint")
    parser.add_argument("--p2", choices=["checkpoint", "scripted", "random"], default="scripted")
    parser.add_argument("--checkpoint", default="checkpoints/sim_linear_policy.npz")
    parser.add_argument("--p2-checkpoint", default=None)
    parser.add_argument("--p1-scripted", default="poke", choices=sorted(SCRIPTED_POLICIES))
    parser.add_argument("--p2-scripted", default="rushdown", choices=sorted(SCRIPTED_POLICIES))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--headless-steps", type=int, default=0)
    args = parser.parse_args()

    env = TekkenLiteEnv(seed=args.seed)
    p1 = build_policy(args.p1, args.checkpoint, args.p1_scripted)
    p2 = build_policy(args.p2, args.p2_checkpoint or args.checkpoint, args.p2_scripted)
    if args.headless_steps > 0:
        run_headless(env, p1, p2, args.headless_steps)
        return 0

    app = SimVisualizer(env=env, p1=p1, p2=p2, steps_per_tick=args.speed)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

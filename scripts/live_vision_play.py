from __future__ import annotations

import argparse
import time

import torch

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.io.screen_backend import DxcamScreenStateBackend
from t8_agent.live.vision_agent import LiveVisionAgent, LiveVisualPpoAgent
from t8_agent.vision.temporal import TemporalScreenEstimator
from t8_agent.vision.model import LearnedTemporalEstimator
from t8_agent.sim.actions import SimAction


class ActionCommitmentFilter:
    DURATIONS = {
        SimAction.JAB: 0.28,
        SimAction.DF1: 0.40,
        SimAction.F2: 0.52,
        SimAction.DB3: 0.55,
        SimAction.HOPKICK: 0.65,
        SimAction.THROW: 0.50,
    }

    def __init__(self) -> None:
        self.committed_until = 0.0

    def filter(self, action: SimAction) -> SimAction:
        now = time.perf_counter()
        if now < self.committed_until:
            return SimAction.NEUTRAL
        duration = self.DURATIONS.get(action)
        if duration is not None:
            self.committed_until = now + duration
        return action


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU V2 policy and screen-only Tekken live runtime.")
    parser.add_argument("--screen-config", default="config/live_screen.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=0.035)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda",
                        help="Torch device for the vision model and V2 policy (default: cuda).")
    parser.add_argument("--agent", choices=["rule", "v2"], default="rule")
    parser.add_argument("--ppo-checkpoint", default=None,
                        help="Native .t8ppo checkpoint trained with --observation-mode visual.")
    parser.add_argument("--player", type=int, choices=[1, 2], default=1,
                        help="Which HUD player the policy controls (default: 1).")
    parser.add_argument("--opponent-character", default=None,
                        help="Roster slug required by a 95-feature matchup checkpoint (for example reina).")
    parser.add_argument("--opponent-archetype", default=None,
                        help="Archetype required by a 95-feature checkpoint (for example rushdown or movement_specialist).")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions; live V2 inference is deterministic by default.")
    parser.add_argument("--hotkey", default="f8", help="Global hotkey used to pause/resume controller output.")
    parser.add_argument("--side-hotkey", default="f7", help="Global hotkey used to flip left/right directional inputs.")
    parser.add_argument("--facing", type=int, choices=[-1, 1], default=1)
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument("--start-paused", dest="start_paused", action="store_true",
                             help="Start with controller output paused (default).")
    start_group.add_argument("--start-enabled", dest="start_paused", action="store_false",
                             help="Opt in to controller output immediately on launch.")
    parser.set_defaults(start_paused=True)
    args = parser.parse_args()

    try:
        import keyboard
    except ImportError as exc:
        raise SystemExit(
            "Missing keyboard module. Install live extras with: "
            '.\\.venv\\Scripts\\python -m pip install -e ".[live]"'
        ) from exc

    backend = DxcamScreenStateBackend(config_path=args.screen_config)
    estimator = TemporalScreenEstimator(
        p1_region=backend.p1_body_region,
        p2_region=backend.p2_body_region,
        motion_threshold=float(backend.config.get("motion_threshold", 0.015)),
    )
    learned_estimator = LearnedTemporalEstimator(args.model, device=args.device) if args.model else None
    if args.agent == "v2" and not args.ppo_checkpoint:
        parser.error("--ppo-checkpoint is required with --agent v2")
    agent = (LiveVisualPpoAgent(args.ppo_checkpoint, device=args.device,
                               deterministic=not args.stochastic, player=args.player,
                               opponent_character=args.opponent_character,
                               opponent_archetype=args.opponent_archetype)
             if args.agent == "v2" else LiveVisionAgent())
    controller = None if args.dry_run else VGamepadInputBackend(facing=args.facing, tap_seconds=0.035)
    commitment = ActionCommitmentFilter()
    enabled = not args.start_paused

    def toggle() -> None:
        nonlocal enabled
        enabled = not enabled
        if not enabled and controller is not None:
            controller.release_all()
        print(f"controller={'on' if enabled else 'off'}", flush=True)

    def flip_side() -> None:
        if controller is None:
            return
        facing = controller.flip_facing()
        print(f"facing={'right' if facing == 1 else 'left'}", flush=True)

    keyboard.add_hotkey(args.hotkey, toggle)
    keyboard.add_hotkey(args.side_hotkey, flip_side)
    print(
        f"controller_hotkey={args.hotkey.upper()} side_hotkey={args.side_hotkey.upper()} "
        f"initial={'off' if args.start_paused else 'on'} facing={'right' if args.facing == 1 else 'left'}",
        flush=True,
    )
    deadline = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < deadline:
            state = backend.read()
            if backend.last_frame is None:
                continue
            estimate = learned_estimator.update(backend.last_frame) if learned_estimator else estimator.update(state, backend.last_frame)
            if estimate is None:
                continue
            policy_action = agent.act(estimate)
            action = commitment.filter(policy_action) if enabled else SimAction.NEUTRAL
            if controller is not None and enabled:
                controller.send_action(action)
            print(
                f"policy={policy_action.value:<12} sent={action.value:<12} distance={estimate.distance:4.2f} "
                f"controller={'on' if enabled else 'off'} "
                f"p1_motion={estimate.p1_motion:.3f} p2_motion={estimate.p2_motion:.3f} "
                f"p2_attack={estimate.p2_attack_likelihood:.2f}",
                flush=True,
            )
            time.sleep(args.interval)
    finally:
        if controller is not None:
            controller.release_all()
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

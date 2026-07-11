from __future__ import annotations

import argparse
import time
from pathlib import Path

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.io.screen_backend import DxcamScreenStateBackend
from t8_agent.live.agents import LivePpoCheckpointAgent, LiveScriptedAgent, find_latest_checkpoint
from t8_agent.sim.tekken_lite import SimAction


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline/local Tekken live test runner with hotkey toggle.")
    parser.add_argument("--screen-config", default="config/live_screen.example.yaml")
    parser.add_argument("--hotkey", default="f8")
    parser.add_argument("--quit-hotkey", default="f12")
    parser.add_argument("--interval", type=float, default=0.12)
    parser.add_argument("--tap-seconds", type=float, default=0.10)
    parser.add_argument("--dash-gap-seconds", type=float, default=0.035)
    parser.add_argument("--facing", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--lp-button", choices=["x", "y", "a", "b"], default="x", help="Xbox button mapped to Tekken 1.")
    parser.add_argument("--rp-button", choices=["x", "y", "a", "b"], default="y", help="Xbox button mapped to Tekken 2.")
    parser.add_argument("--lk-button", choices=["x", "y", "a", "b"], default="a", help="Xbox button mapped to Tekken 3.")
    parser.add_argument("--rk-button", choices=["x", "y", "a", "b"], default="b", help="Xbox button mapped to Tekken 4.")
    parser.add_argument("--agent", choices=["checkpoint", "scripted"], default="checkpoint")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PPO actions. Live testing defaults to stochastic.")
    parser.add_argument("--no-unstick-filter", action="store_true", help="Disable the live anti-crouch action filter.")
    parser.add_argument("--max-non-attack-streak", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Capture screen and print actions without pressing gamepad.")
    parser.add_argument("--self-test", action="store_true", help="Capture one frame, print status, and exit.")
    args = parser.parse_args()

    screen_config = Path(args.screen_config)
    state_backend = DxcamScreenStateBackend(config_path=screen_config if screen_config.exists() else None)
    if args.self_test:
        state = state_backend.read()
        _print_status(SimAction.NEUTRAL, state)
        state_backend.close()
        return 0

    try:
        import keyboard
    except ImportError as exc:
        raise SystemExit(
            "Missing keyboard module. Install live extras with: "
            '.\\.venv\\Scripts\\python -m pip install -e ".[live]"'
        ) from exc

    controller = None if args.dry_run else VGamepadInputBackend(
        facing=args.facing,
        tap_seconds=args.tap_seconds,
        dash_gap_seconds=args.dash_gap_seconds,
        lp_button=args.lp_button,
        rp_button=args.rp_button,
        lk_button=args.lk_button,
        rk_button=args.rk_button,
    )
    if args.agent == "checkpoint":
        checkpoint = find_latest_checkpoint() if args.checkpoint == "latest" else Path(args.checkpoint)
        agent = LivePpoCheckpointAgent(checkpoint, deterministic=args.deterministic)
        print(f"loaded_checkpoint={checkpoint}", flush=True)
        print(f"checkpoint_interface={agent.compatibility_label}", flush=True)
    else:
        agent = LiveScriptedAgent(seed=8080)
    action_filter = LiveActionFilter(enabled=not args.no_unstick_filter, max_non_attack_streak=args.max_non_attack_streak)
    enabled = False

    def toggle() -> None:
        nonlocal enabled
        enabled = not enabled
        if not enabled and controller is not None:
            controller.release_all()
        print(f"{'enabled' if enabled else 'paused'}", flush=True)

    keyboard.add_hotkey(args.hotkey, toggle)
    print("Offline/local live runner ready.", flush=True)
    print(f"Press {args.hotkey.upper()} to start/pause. Press {args.quit_hotkey.upper()} to quit.", flush=True)
    print("Keep this in Practice/Offline mode only. The controller is released whenever paused.", flush=True)
    try:
        while not keyboard.is_pressed(args.quit_hotkey):
            state = state_backend.read()
            if enabled:
                if (state.raw or {}).get("capture_valid") is False:
                    if controller is not None:
                        controller.release_all()
                    _print_status(SimAction.NEUTRAL, state)
                    time.sleep(args.interval)
                    continue
                action = agent.act(state)
                action = action_filter.filter(action)
                if controller is not None:
                    controller.send_action(action)
                _print_status(action, state)
            time.sleep(args.interval)
    finally:
        if controller is not None:
            controller.release_all()
        state_backend.close()
    print("stopped", flush=True)
    return 0


class LiveActionFilter:
    ATTACKS = {
        SimAction.JAB,
        SimAction.DF1,
        SimAction.F2,
        SimAction.DB3,
        SimAction.HOPKICK,
        SimAction.THROW,
    }

    def __init__(self, *, enabled: bool, max_non_attack_streak: int) -> None:
        self.enabled = enabled
        self.max_non_attack_streak = max(1, max_non_attack_streak)
        self.low_block_streak = 0
        self.neutral_streak = 0
        self.non_attack_streak = 0
        self.replacements = [
            SimAction.WALK_FORWARD,
            SimAction.JAB,
            SimAction.DF1,
            SimAction.F2,
            SimAction.DB3,
            SimAction.THROW,
            SimAction.HOPKICK,
        ]
        self.replacement_idx = 0

    def filter(self, action: SimAction) -> SimAction:
        if not self.enabled:
            return action
        if action == SimAction.BLOCK_LOW:
            self.low_block_streak += 1
        else:
            self.low_block_streak = 0
        if action == SimAction.NEUTRAL:
            self.neutral_streak += 1
        else:
            self.neutral_streak = 0
        if action in self.ATTACKS:
            self.non_attack_streak = 0
        else:
            self.non_attack_streak += 1
        if (
            self.low_block_streak < 3
            and self.neutral_streak < 4
            and self.non_attack_streak < self.max_non_attack_streak
        ):
            return action
        replacement = self.replacements[self.replacement_idx % len(self.replacements)]
        self.replacement_idx += 1
        self.low_block_streak = 0
        self.neutral_streak = 0
        self.non_attack_streak = 0
        return replacement


def _print_status(action: SimAction, state) -> None:
    raw = state.raw or {}
    print(
        f"action={action.value:<12} "
        f"p1_hp={state.p1.health:6.1f} p2_hp={state.p2.health:6.1f} "
        f"screen={raw.get('screen_width', '?')}x{raw.get('screen_height', '?')}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

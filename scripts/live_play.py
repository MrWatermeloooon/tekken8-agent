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
    parser.add_argument("--tap-seconds", type=float, default=0.055)
    parser.add_argument("--facing", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--agent", choices=["checkpoint", "scripted"], default="checkpoint")
    parser.add_argument("--checkpoint", default="latest")
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

    controller = None if args.dry_run else VGamepadInputBackend(facing=args.facing, tap_seconds=args.tap_seconds)
    if args.agent == "checkpoint":
        checkpoint = find_latest_checkpoint() if args.checkpoint == "latest" else Path(args.checkpoint)
        agent = LivePpoCheckpointAgent(checkpoint)
        print(f"loaded_checkpoint={checkpoint}")
    else:
        agent = LiveScriptedAgent(seed=8080)
    enabled = False

    def toggle() -> None:
        nonlocal enabled
        enabled = not enabled
        if not enabled and controller is not None:
            controller.release_all()
        print(f"{'enabled' if enabled else 'paused'}")

    keyboard.add_hotkey(args.hotkey, toggle)
    print("Offline/local live runner ready.")
    print(f"Press {args.hotkey.upper()} to start/pause. Press {args.quit_hotkey.upper()} to quit.")
    print("Keep this in Practice/Offline mode only. The controller is released whenever paused.")
    try:
        while not keyboard.is_pressed(args.quit_hotkey):
            state = state_backend.read()
            if enabled:
                action = agent.act(state)
                if controller is not None:
                    controller.send_action(action)
                _print_status(action, state)
            time.sleep(args.interval)
    finally:
        if controller is not None:
            controller.release_all()
        state_backend.close()
    print("stopped")
    return 0


def _print_status(action: SimAction, state) -> None:
    raw = state.raw or {}
    print(
        f"action={action.value:<12} "
        f"p1_hp={state.p1.health:6.1f} p2_hp={state.p2.health:6.1f} "
        f"screen={raw.get('screen_width', '?')}x{raw.get('screen_height', '?')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

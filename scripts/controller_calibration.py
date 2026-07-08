from __future__ import annotations

import argparse
import time

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.sim.tekken_lite import SimAction


DEFAULT_SEQUENCE = [
    SimAction.WALK_FORWARD,
    SimAction.WALK_BACK,
    SimAction.BLOCK_HIGH,
    SimAction.BLOCK_LOW,
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Press a known Tekken-lite action sequence for controller calibration.")
    parser.add_argument("--facing", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--tap-seconds", type=float, default=0.12)
    parser.add_argument("--between-seconds", type=float, default=0.75)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--actions", nargs="+", default=[action.value for action in DEFAULT_SEQUENCE])
    args = parser.parse_args()

    controller = VGamepadInputBackend(facing=args.facing, tap_seconds=args.tap_seconds)
    try:
        print(f"Starting in {args.start_delay:.1f}s. Focus Tekken now.", flush=True)
        time.sleep(args.start_delay)
        for action_value in args.actions:
            action = SimAction(action_value)
            print(f"pressing={action.value}", flush=True)
            controller.send_action(action)
            time.sleep(args.between_seconds)
    finally:
        controller.release_all()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import time

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.sim.tekken_lite import SimAction


DEFAULT_SEQUENCE = [
    SimAction.WALK_FORWARD,
    SimAction.WALK_BACK,
    SimAction.DASH_FORWARD,
    SimAction.DASH_BACK,
    SimAction.SIDESTEP_LEFT,
    SimAction.SIDESTEP_RIGHT,
    SimAction.BLOCK_HIGH,
    SimAction.BLOCK_LOW,
    SimAction.LOW_PARRY,
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
    SimAction.THROW_BREAK_1,
    SimAction.THROW_BREAK_2,
    SimAction.THROW_BREAK_1_2,
    SimAction.HEAT_BURST,
    SimAction.RAGE_ART,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Press a known Tekken-lite action sequence for controller calibration.")
    parser.add_argument("--facing", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--tap-seconds", type=float, default=0.12)
    parser.add_argument("--dash-gap-seconds", type=float, default=0.035)
    parser.add_argument("--between-seconds", type=float, default=0.75)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--lp-button", choices=["x", "y", "a", "b"], default="x", help="Xbox button mapped to Tekken 1.")
    parser.add_argument("--rp-button", choices=["x", "y", "a", "b"], default="y", help="Xbox button mapped to Tekken 2.")
    parser.add_argument("--lk-button", choices=["x", "y", "a", "b"], default="a", help="Xbox button mapped to Tekken 3.")
    parser.add_argument("--rk-button", choices=["x", "y", "a", "b"], default="b", help="Xbox button mapped to Tekken 4.")
    parser.add_argument("--actions", nargs="+", default=[action.value for action in DEFAULT_SEQUENCE])
    args = parser.parse_args()

    controller = VGamepadInputBackend(
        facing=args.facing,
        tap_seconds=args.tap_seconds,
        dash_gap_seconds=args.dash_gap_seconds,
        lp_button=args.lp_button,
        rp_button=args.rp_button,
        lk_button=args.lk_button,
        rk_button=args.rk_button,
    )
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

from __future__ import annotations

import argparse
import time

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.sim.actions import SimAction


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
    parser = argparse.ArgumentParser(description="Send a known V2 action sequence for offline controller calibration.")
    parser.add_argument(
        "--confirm-offline",
        action="store_true",
        help="Required acknowledgment that Tekken is in an offline Practice/Versus mode.",
    )
    parser.add_argument("--facing", type=int, default=1, choices=[-1, 1])
    parser.add_argument("--tap-seconds", type=float, default=0.12)
    parser.add_argument("--dash-gap-seconds", type=float, default=0.035)
    parser.add_argument("--between-seconds", type=float, default=0.75)
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--lp-button", choices=["x", "y", "a", "b"], default="x")
    parser.add_argument("--rp-button", choices=["x", "y", "a", "b"], default="y")
    parser.add_argument("--lk-button", choices=["x", "y", "a", "b"], default="a")
    parser.add_argument("--rk-button", choices=["x", "y", "a", "b"], default="b")
    parser.add_argument("--actions", nargs="+", default=[action.value for action in DEFAULT_SEQUENCE])
    args = parser.parse_args()
    if not args.confirm_offline:
        parser.error("--confirm-offline is required before this script can send controller input")

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
        print(f"Starting in {args.start_delay:.1f}s. Focus offline Tekken 8 now.", flush=True)
        time.sleep(args.start_delay)
        for action_value in args.actions:
            action = SimAction(action_value)
            print(f"pressing={action.value}", flush=True)
            controller.send_action(action)
            time.sleep(args.between_seconds)
    finally:
        controller.release_all()
    print("controller_calibration_ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

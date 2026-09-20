from __future__ import annotations

import argparse
import time
import threading

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.io.screen_backend import DxcamScreenStateBackend
from t8_agent.live.vision_agent import LiveVisionAgent, LiveVisualPpoAgent
from t8_agent.live.runtime import LiveActionState, RoundTracker
from t8_agent.vision.temporal import TemporalScreenEstimator
from t8_agent.sim.actions import SimAction


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU V2 policy and screen-only Tekken live runtime.")
    parser.add_argument("--screen-config", default="config/live_screen.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=4.0 / 60.0,
                        help="Decision period in seconds; default matches four simulator frames.")
    parser.add_argument("--capture-timeout", type=float, default=0.25)
    parser.add_argument("--reset-hotkey", default="f6", help="Reset live history after a practice reset.")
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
    parser.add_argument(
        "--allow-automatic-calibration",
        action="store_true",
        help="Allow controller mode with automatic screen regions; interactive calibration is safer.",
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument("--start-paused", dest="start_paused", action="store_true",
                             help="Start with controller output paused (default).")
    start_group.add_argument("--start-enabled", dest="start_paused", action="store_false",
                             help="Opt in to controller output immediately on launch.")
    parser.set_defaults(start_paused=True)
    args = parser.parse_args()
    if not (0 < args.interval < args.capture_timeout < float("inf")) or not (0 < args.seconds < float("inf")):
        parser.error("require 0 < interval < capture-timeout and positive finite seconds")
    if args.agent == "v2" and not args.ppo_checkpoint:
        parser.error("--ppo-checkpoint is required with --agent v2")

    def p1_is_on_left(controlled_facing: int) -> bool:
        controlled_player_is_left = controlled_facing == 1
        return controlled_player_is_left if args.player == 1 else not controlled_player_is_left

    try:
        import keyboard
    except ImportError as exc:
        raise SystemExit(
            "Missing keyboard module. Install live extras with: "
            '.\\.venv\\Scripts\\python -m pip install -e ".[live]"'
        ) from exc

    backend = DxcamScreenStateBackend(
        config_path=args.screen_config,
        p1_on_left=p1_is_on_left(args.facing),
    )
    calibration_source = str(backend.config.get("calibration_source", "unspecified"))
    has_regions = all(
        region is not None
        for region in (
            backend.p1_health_region,
            backend.p2_health_region,
            backend.p1_body_region,
            backend.p2_body_region,
        )
    )
    unsafe_calibration = calibration_source.startswith("automatic") or calibration_source in {
        "unspecified",
        "uncalibrated_template",
    }
    if not args.dry_run and (not has_regions or unsafe_calibration) and not args.allow_automatic_calibration:
        backend.close()
        parser.error(
            "controller mode requires interactive screen calibration; run "
            "scripts/calibrate_live_screen.py or explicitly pass --allow-automatic-calibration"
        )
    estimator = TemporalScreenEstimator(
        p1_region=backend.p1_body_region,
        p2_region=backend.p2_body_region,
        motion_threshold=float(backend.config.get("motion_threshold", 0.015)),
        p1_on_left=p1_is_on_left(args.facing),
    )
    if args.model:
        from t8_agent.vision.model import LearnedTemporalEstimator
        learned_estimator = LearnedTemporalEstimator(args.model, device=args.device)
    else:
        learned_estimator = None
    agent = (LiveVisualPpoAgent(args.ppo_checkpoint, device=args.device,
                               deterministic=not args.stochastic, player=args.player,
                               opponent_character=args.opponent_character,
                               opponent_archetype=args.opponent_archetype)
             if args.agent == "v2" else LiveVisionAgent())
    controller = None if args.dry_run else VGamepadInputBackend(
        facing=args.facing, tap_seconds=0.035, asynchronous=True, hold_timeout=args.capture_timeout)
    action_state = LiveActionState(player=args.player)
    rounds = RoundTracker()
    enabled = not args.start_paused
    current_facing = args.facing
    control_lock = threading.RLock()
    reset_requested = threading.Event()
    side_flips = 0
    if controller is not None:
        controller.set_enabled(enabled)

    def reset_history() -> None:
        agent.reset_episode()
        estimator.reset_episode()
        if learned_estimator is not None:
            learned_estimator.reset_episode()
        action_state.reset_episode()

    def pause_for_capture() -> None:
        nonlocal enabled
        with control_lock:
            if enabled:
                print("controller=off reason=stale_capture; press the enable hotkey after capture recovers", flush=True)
            enabled = False
            if controller is not None:
                controller.set_enabled(False)
            reset_requested.set()

    def toggle() -> None:
        nonlocal enabled
        with control_lock:
            enabled = not (enabled and (controller is None or controller.enabled))
            if controller is not None:
                controller.set_enabled(enabled)
            reset_requested.set()
            print(f"controller={'on' if enabled else 'off'}", flush=True)

    def flip_side() -> None:
        nonlocal side_flips
        with control_lock:
            side_flips += 1
            if controller is not None:
                controller.release_all()
            reset_requested.set()

    def request_reset() -> None:
        with control_lock:
            if controller is not None:
                controller.release_all()
            reset_requested.set()

    print(
        f"controller_hotkey={args.hotkey.upper()} side_hotkey={args.side_hotkey.upper()} "
        f"reset_hotkey={args.reset_hotkey.upper()} "
        f"initial={'off' if args.start_paused else 'on'} facing={'right' if args.facing == 1 else 'left'}",
        flush=True,
    )
    deadline = time.perf_counter() + args.seconds
    hotkeys = []
    next_decision = time.perf_counter()
    try:
        hotkeys.append(keyboard.add_hotkey(args.hotkey, toggle))
        hotkeys.append(keyboard.add_hotkey(args.side_hotkey, flip_side))
        hotkeys.append(keyboard.add_hotkey(args.reset_hotkey, request_reset))
        if args.agent == "v2" and agent.observation_size == 95:
            print("opponent_move_context=unknown (-1); existing oracle-trained checkpoints require transfer validation", flush=True)
        while time.perf_counter() < deadline:
            time.sleep(max(0.0, next_decision - time.perf_counter()))
            if time.perf_counter() >= deadline:
                break
            next_decision = time.perf_counter() + args.interval
            with control_lock:
                if reset_requested.is_set():
                    if side_flips % 2:
                        current_facing *= -1
                        if controller is not None:
                            controller.flip_facing()
                        backend.set_p1_on_left(p1_is_on_left(current_facing))
                        estimator.set_p1_on_left(p1_is_on_left(current_facing))
                        print(f"facing={'right' if current_facing == 1 else 'left'}", flush=True)
                    side_flips = 0
                    reset_history()
                    rounds.reset_episode()
                    reset_requested.clear()
                generation = controller.generation if controller is not None else None
            state = backend.read()
            raw = state.raw or {}
            if not raw.get("capture_fresh", False):
                if raw.get("frame_age_seconds", float("inf")) >= args.capture_timeout:
                    pause_for_capture()
                continue
            with control_lock:
                if controller is not None:
                    if enabled and not controller.enabled:
                        pause_for_capture()
                    controller.heartbeat()
            round_status = rounds.update(raw["p1_health_ratio"], raw["p2_health_ratio"])
            if round_status == "waiting":
                if controller is not None:
                    controller.release_all()
                continue
            if round_status == "reset":
                reset_history()
                if controller is not None:
                    controller.release_all()
                    generation = controller.generation
            estimate = learned_estimator.update(backend.last_frame) if learned_estimator else estimator.update(state, backend.last_frame)
            if estimate is None:
                continue
            input_busy = controller.is_busy if controller is not None else False
            mask = action_state.action_mask(estimate, input_busy=input_busy)
            policy_action = agent.act(estimate, action_mask=mask)
            action = SimAction.NEUTRAL
            with control_lock:
                if time.perf_counter() - backend.last_capture_at >= args.capture_timeout:
                    pause_for_capture()
                elif enabled and not reset_requested.is_set() and not input_busy:
                    if controller is None or controller.send_action(policy_action, expected_generation=generation):
                        action = policy_action
                        action_state.record_executed(action)
            print(
                f"policy={policy_action.value:<12} sent={action.value:<12} distance={estimate.distance:4.2f} "
                f"controller={'on' if enabled else 'off'} "
                f"p1_motion={estimate.p1_motion:.3f} p2_motion={estimate.p2_motion:.3f} "
                f"p2_attack={estimate.p2_attack_likelihood:.2f}",
                flush=True,
            )
    finally:
        try:
            if controller is not None:
                controller.close()
        finally:
            try:
                backend.close()
            finally:
                for hotkey in hotkeys:
                    keyboard.remove_hotkey(hotkey)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

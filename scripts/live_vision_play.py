from __future__ import annotations

import argparse
import time
import threading

from pathlib import Path

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.live.feedback import ExecutionFeedback, ReplayRecorder, SideSwitchDetector
from t8_agent.io.screen_backend import DxcamScreenStateBackend
from t8_agent.live.vision_agent import LiveVisionAgent, LiveVisualPpoAgent
from t8_agent.live.runtime import LiveActionState, RoundTracker
from t8_agent.vision.temporal import TemporalScreenEstimator
from t8_agent.sim.actions import SimAction


# Compatibility actions whose outcome (hit, interrupted, ...) is tracked.
ATTACK_ACTIONS = frozenset(LiveActionState.RECOVERY_LIMITS) | {
    SimAction.HEAT_BURST, SimAction.HEAT_SMASH, SimAction.RAGE_ART}


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
    parser.add_argument("--agent", choices=["rule", "v2", "full-move"], default="rule")
    parser.add_argument("--ppo-checkpoint", default=None,
                        help="Native .t8ppo checkpoint trained with --observation-mode visual.")
    parser.add_argument("--player", type=int, choices=[1, 2], default=1,
                        help="Which HUD player the policy controls (default: 1).")
    parser.add_argument("--opponent-character", default=None,
                        help="Roster slug required by a 95-feature matchup checkpoint (for example reina).")
    parser.add_argument("--opponent-archetype", default=None,
                        help="Archetype required by a 95-feature checkpoint (for example rushdown or movement_specialist).")
    parser.add_argument("--controlled-character", default=None,
                        help="Roster slug of the fighter this runtime controls; required for --agent full-move.")
    parser.add_argument("--validation-root", default="data/validation")
    parser.add_argument("--allow-unvalidated", action="store_true",
                        help="Dry run only: load a full-move checkpoint whose character gate has not passed.")
    parser.add_argument("--auto-facing", dest="auto_facing", action="store_true", default=True,
                        help="Detect side switches from fighter colors (default; F7 still overrides).")
    parser.add_argument("--no-auto-facing", dest="auto_facing", action="store_false")
    parser.add_argument("--replay", default=None,
                        help="Write a JSONL replay of decisions, controller presses, and outcomes here.")
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
    if args.agent in ("v2", "full-move") and not args.ppo_checkpoint:
        parser.error(f"--ppo-checkpoint is required with --agent {args.agent}")
    if args.agent == "full-move" and not args.controlled_character:
        parser.error("--controlled-character is required with --agent full-move")
    if args.allow_unvalidated and not args.dry_run:
        parser.error("--allow-unvalidated is only allowed with --dry-run")

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
    # Screen-only checkpoints also need this capture setup's measured
    # uncertainty (scripts/calibrate_screen_uncertainty.py writes both keys).
    screen_sigma = backend.config.get("screen_position_sigma")
    screen_error = backend.config.get("screen_event_error")
    agent = (LiveVisualPpoAgent(args.ppo_checkpoint, device=args.device,
                               deterministic=not args.stochastic, player=args.player,
                               opponent_character=args.opponent_character,
                               opponent_archetype=args.opponent_archetype,
                               screen_position_sigma=None if screen_sigma is None else float(screen_sigma),
                               screen_event_error=None if screen_error is None else float(screen_error),
                               motion_threshold=float(backend.config.get("motion_threshold", 0.015)))
             if args.agent == "v2" else LiveVisionAgent())  # full-move: FullMovePolicy below
    full_move = None
    move_catalog = ledger = None
    own_tracker = heat_estimate = rage_estimate = None
    hud_tracker = None
    moves_by_id: dict = {}
    if args.agent == "full-move":
        from t8_agent.live.full_move import FullMovePolicy, verify_full_move_checkpoint
        from t8_agent.moves.catalog import load_compiled_catalog
        from t8_agent.moves.validation import load_validation_ledger
        move_catalog = load_compiled_catalog(Path("data/generated/full_move_catalog.json"))
        checkpoint, table = verify_full_move_checkpoint(
            args.ppo_checkpoint, move_catalog, args.controlled_character,
            validation_root=args.validation_root, allow_unvalidated=args.allow_unvalidated)
        full_move = FullMovePolicy(checkpoint, table, deterministic=not args.stochastic)
        ledger = load_validation_ledger(Path(args.validation_root) / f"{args.controlled_character}.json")
        moves_by_id = {move["stable_id"]: move for move in move_catalog.moves}
        roster_ids = {row["slug"]: int(row["id"]) for row in move_catalog.characters}
        controlled_id = roster_ids[args.controlled_character]
        from t8_agent.live.hud_reader import HudTracker
        hud_tracker = HudTracker()  # health, recoverable health, Heat gauges (screen contract v4)
        # A --live-noise checkpoint also reads this setup's calibrated noise levels: position sigma
        # (m), detection error rate, and input latency (frames); a dry run falls back to mid values.
        full_move_noise = None
        if "noise" in checkpoint.observation_contract:
            latency = backend.config.get("controller_latency_frames")
            if (screen_sigma is None or screen_error is None or latency is None) and not args.dry_run:
                parser.error("a --live-noise checkpoint needs screen_position_sigma, screen_event_error, and "
                             "controller_latency_frames in the screen config")
            full_move_noise = (float(screen_sigma if screen_sigma is not None else 0.12),
                               float(screen_error if screen_error is not None else 0.05),
                               float(latency if latency is not None else 4))
        opponent_id = roster_ids.get(args.opponent_character or "")  # unknown opponent: no one-hot
        from t8_agent.live.own_state import (
            HeatEstimate, HudGauge, OwnStateTracker, RageEstimate, load_move_timings,
        )
        own_tracker = OwnStateTracker(load_move_timings(Path("data/generated/full_combat_bindings.csv"),
                                                        args.controlled_character))
        heat_estimate = HeatEstimate(HudGauge.from_config(backend.config, f"p{args.player}_heat_gauge"))
        rage_estimate = RageEstimate(float(backend.config.get("rage_health_fraction", 0.25)))
        print(f"controlled_character={args.controlled_character} candidates={len(table)} "
              f"own_state=tracked from sent commands, heat="
              f"{'HUD gauge' if heat_estimate.gauge else 'unknown (calibrate p' + str(args.player) + '_heat_gauge)'}, "
              "rage=health threshold (provisional), opponent move=unknown", flush=True)
    recorder = None
    if args.replay:
        recorder = ReplayRecorder(args.replay, metadata={
            "agent": args.agent, "checkpoint": args.ppo_checkpoint, "player": args.player,
            "controlled_character": args.controlled_character, "dry_run": args.dry_run})
    controller = None if args.dry_run else VGamepadInputBackend(
        facing=args.facing, tap_seconds=0.035, asynchronous=True, hold_timeout=args.capture_timeout,
        # String input timing validated in Practice (scripts/validate_moves.py, 2026-10-10).
        string_gap_seconds=14 / 60, string_first_gap_seconds=8 / 60, string_gap_growth_seconds=4 / 60,
        dash_button_delay_seconds=4 / 60,
        input_log=None if recorder is None else recorder.controller)
    feedback = ExecutionFeedback(player=args.player,
                                 motion_threshold=float(backend.config.get("motion_threshold", 0.015)))
    side_detector = SideSwitchDetector() if args.auto_facing else None
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
        if own_tracker is not None:
            own_tracker.reset_round()
            heat_estimate.reset_round()
            hud_tracker.reset_round()
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

    def flip_side(manual: bool = True) -> None:
        nonlocal side_flips
        with control_lock:
            side_flips += 1
            if controller is not None:
                controller.release_all()
            if manual and side_detector is not None:
                # The manual override wins; colors are re-learned from the next frame.
                side_detector.reset_reference()
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
            # No new frame yet (about half of the grabs): poll again shortly instead of
            # skipping this decision and sleeping a whole interval.
            retry_until = time.perf_counter() + args.interval
            while not raw.get("capture_fresh", False) and time.perf_counter() < retry_until:
                time.sleep(0.002)
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
            frame = backend.last_frame
            left, right = backend.p1_body_region, backend.p2_body_region
            if side_detector is not None and frame is not None and left is not None and right is not None:
                if side_detector.update(frame[left.top:left.bottom, left.left:left.right],
                                        frame[right.top:right.bottom, right.left:right.right]):
                    print("side_switch=detected", flush=True)
                    flip_side(manual=False)
                    continue
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
            resolved = feedback.update(estimate)
            if resolved is not None and recorder is not None:
                recorder.outcome(resolved)
            input_busy = controller.is_busy if controller is not None else False
            if full_move is not None:
                from t8_agent.live.full_move import dispatch_candidate, live_legal_mask
                from t8_agent.live.own_state import screen_full_move_observation
                now = time.perf_counter()
                if (estimate.p1_hit_event if args.player == 1 else estimate.p2_hit_event):
                    own_tracker.on_own_hit(now)
                if resolved is not None and resolved.outcome == "hit":
                    own_tracker.on_outcome("hit")
                own_health = estimate.p1_health_ratio if args.player == 1 else estimate.p2_health_ratio
                observed = own_tracker.observe(
                    now, busy=input_busy,
                    heat=heat_estimate.update(frame) if frame is not None else None,
                    rage=rage_estimate.update(own_health))
                candidate_mask = live_legal_mask(full_move.table, move_catalog, observed,
                                                 ledger=ledger, require_validated=not args.allow_unvalidated)
                own_key, other_key = ("p1_posture", "p2_posture") if args.player == 1 else ("p2_posture", "p1_posture")
                choice = full_move.act(screen_full_move_observation(
                    estimate, observed, player=args.player, own_character=controlled_id,
                    opponent_character=opponent_id,
                    attack_cue_threshold=float(backend.config.get("attack_cue_threshold", 0.5)),
                    noise_levels=full_move_noise, own_pose=raw.get(own_key), opponent_pose=raw.get(other_key),
                    own_move=own_tracker.move_progress(now),
                    hud=hud_tracker.update(backend.last_frame) if backend.last_frame is not None else None),
                    candidate_mask)
                candidate = full_move.table.stable_ids[choice]
                legal_count = int(candidate_mask.sum())
            else:
                mask = action_state.action_mask(estimate, input_busy=input_busy)
                policy_action = agent.act(estimate, action_mask=mask)
                candidate = policy_action.value
                legal_count = int(mask.sum())
            action_value = SimAction.NEUTRAL.value
            sent = False
            attempted = False
            with control_lock:
                if time.perf_counter() - backend.last_capture_at >= args.capture_timeout:
                    pause_for_capture()
                elif enabled and not reset_requested.is_set() and not input_busy:
                    attempted = True
                    if full_move is not None:
                        sent = controller is None or dispatch_candidate(
                            controller, full_move.table, choice, expected_generation=generation)
                        if sent:
                            own_tracker.on_dispatch(candidate, time.perf_counter())
                    else:
                        sent = controller is None or controller.send_action(
                            policy_action, expected_generation=generation)
                        if sent:
                            action_state.record_executed(policy_action)
                    if sent:
                        action_value = candidate
            expected_inputs = None
            if sent and controller is not None:
                if full_move is not None and not full_move.table.is_universal(choice):
                    expected_inputs = controller.planned_inputs(
                        command=full_move.table.commands[choice], move_id=candidate)
                else:
                    expected_inputs = controller.planned_inputs(action=candidate.split(":")[-1])
            is_attack = (not full_move.table.is_universal(choice) if full_move is not None else
                         policy_action in ATTACK_ACTIONS)
            if attempted and is_attack:
                startup = (moves_by_id.get(candidate) or {}).get("startup")
                feedback.start(candidate, sent=bool(sent), startup=startup)
            if recorder is not None:
                recorder.decision(estimate=estimate, candidate=candidate, sent=bool(sent),
                                  legal_count=legal_count, expected_inputs=expected_inputs)
            print(
                f"policy={candidate:<12} sent={action_value:<12} distance={estimate.distance:4.2f} "
                f"controller={'on' if enabled else 'off'} "
                f"p1_motion={estimate.p1_motion:.3f} p2_motion={estimate.p2_motion:.3f} "
                f"p2_attack={estimate.p2_attack_likelihood:.2f}",
                flush=True,
            )
    finally:
        if recorder is not None:
            recorder.write("summary", outcomes=feedback.summary())
            recorder.close()
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

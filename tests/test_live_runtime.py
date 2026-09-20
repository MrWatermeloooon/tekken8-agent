from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

import numpy as np

from t8_agent.live.runtime import LiveActionState, RoundTracker
from t8_agent.sim.action_space import ACTION_SPACE
from t8_agent.sim.actions import SimAction
from t8_agent.vision.temporal import VisualEstimate


def estimate(**changes):
    base = VisualEstimate(1.0, 1.0, -0.85, 0.85, 1.7, 0.0, 0.0, 0.0, 0.0,
                          False, False, 0.0, 0.0)
    return replace(base, **changes)


def legal(mask, action):
    return mask[ACTION_SPACE.index(action)]


def test_recovery_masks_attacks_but_retains_defense_and_uses_motion_feedback():
    state = LiveActionState()
    state.record_executed(SimAction.HOPKICK, now=10.0)
    mask = state.action_mask(estimate(p1_motion=0.1), now=10.1)
    assert not legal(mask, SimAction.JAB)
    assert legal(mask, SimAction.BLOCK_LOW)
    assert legal(mask, SimAction.THROW_BREAK_1)
    state.action_mask(estimate(), now=10.2)
    assert legal(state.action_mask(estimate(), now=10.3), SimAction.JAB)


def test_hit_clears_old_recovery_for_both_player_sides():
    for player in (1, 2):
        state = LiveActionState(player)
        state.record_executed(SimAction.HOPKICK, now=10.0)
        hit = estimate(**{f"p{player}_hit_event": True})
        mask = state.action_mask(hit, now=10.1)
        assert not legal(mask, SimAction.JAB)
        assert legal(mask, SimAction.BLOCK_HIGH)
        assert legal(state.action_mask(estimate(), now=10.2), SimAction.JAB)


def test_unobserved_recovery_expires_and_pending_input_has_single_choice():
    state = LiveActionState()
    state.record_executed(SimAction.F2, now=10.0)
    assert not legal(state.action_mask(estimate(), now=10.2), SimAction.JAB)
    assert legal(state.action_mask(estimate(), now=11.0), SimAction.JAB)
    mask = state.action_mask(estimate(), input_busy=True, now=11.0)
    assert mask.sum() == 1 and legal(mask, SimAction.NEUTRAL)


def test_round_reset_is_debounced_and_ko_waits_for_health_restoration():
    rounds = RoundTracker()
    assert [rounds.update(1, 1) for _ in range(3)] == ["waiting", "waiting", "reset"]
    assert rounds.update(0.5, 0.3) == "active"
    assert rounds.update(0.5, 0) == "waiting"
    assert rounds.update(0.5, 0) == "waiting"
    assert rounds.update(0.5, 0.3) == "waiting"
    assert [rounds.update(1, 1) for _ in range(3)] == ["waiting", "waiting", "reset"]
    assert rounds.update(1, 1) == "active"


def test_practice_reset_and_single_bad_health_frame():
    rounds = RoundTracker()
    for _ in range(3):
        rounds.update(1, 1)
    rounds.update(0.7, 0.8)
    assert rounds.update(1, 1) == "waiting"
    assert rounds.update(0.7, 0.8) == "active"
    assert rounds.update(np.nan, 1) == "waiting"
    assert [rounds.update(1, 1) for _ in range(3)][-1] == "reset"


def run_live_loop(monkeypatch, samples, on_action=None):
    from t8_agent.core.types import GameState, PlayerState
    from t8_agent.io.controller_backend import VGamepadInputBackend

    script = Path(__file__).parents[1] / "scripts" / "live_vision_play.py"
    spec = importlib.util.spec_from_file_location("live_runtime_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [10.0]
    callbacks = {}
    removed = []

    def sleep(duration):
        clock[0] += duration

    def add_hotkey(key, callback):
        callbacks[key] = callback
        return key

    monkeypatch.setitem(sys.modules, "keyboard", SimpleNamespace(
        add_hotkey=add_hotkey, remove_hotkey=removed.append))
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock[0], sleep=sleep))

    class Source:
        config = {"calibration_source": "interactive"}
        p1_health_region = p2_health_region = p1_body_region = p2_body_region = object()
        last_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        last_capture_at = 10.0
        reads = 0
        closed = False

        def read(self):
            h1, h2, fresh = samples[min(self.reads, len(samples) - 1)]
            self.reads += 1
            if fresh:
                self.last_capture_at = clock[0]
            return GameState(PlayerState(h1 * 180, -1), PlayerState(h2 * 180, 1), 60,
                             raw={"capture_fresh": fresh,
                                  "frame_age_seconds": clock[0] - self.last_capture_at,
                                  "p1_health_ratio": h1, "p2_health_ratio": h2})

        def close(self):
            self.closed = True

    class Estimator:
        resets = 0

        def update(self, state, frame):
            return estimate(p1_health_ratio=state.p1.health / 180, p2_health_ratio=state.p2.health / 180)

        def reset_episode(self):
            self.resets += 1

    class Agent:
        observation_size = 95
        calls = 0
        resets = 0

        def act(self, observation, *, action_mask):
            self.calls += 1
            assert legal(action_mask, SimAction.BLOCK_LOW)
            if on_action is not None:
                on_action(self.calls, callbacks)
            return SimAction.BLOCK_LOW

        def reset_episode(self):
            self.resets += 1

    class Pad:
        held = set()
        presses = 0

        def press_button(self, *, button):
            self.held.add(button)
            self.presses += 1

        def release_button(self, *, button):
            self.held.discard(button)

        def update(self):
            pass

    source, estimator, agent, pad = Source(), Estimator(), Agent(), Pad()
    controller = VGamepadInputBackend(gamepad=pad)
    monkeypatch.setattr(module, "DxcamScreenStateBackend", lambda **kw: source)
    monkeypatch.setattr(module, "TemporalScreenEstimator", lambda **kw: estimator)
    monkeypatch.setattr(module, "LiveVisualPpoAgent", lambda *args, **kw: agent)
    monkeypatch.setattr(module, "VGamepadInputBackend", lambda **kw: controller)
    monkeypatch.setattr(sys, "argv", [str(script), "--agent", "v2", "--ppo-checkpoint", "unused",
                                     "--start-enabled", "--interval", "0.05", "--capture-timeout", "0.2",
                                     "--seconds", str(len(samples) * 0.05 - 0.001)])
    assert module.main() == 0
    assert source.closed and not pad.held
    assert set(removed) == {"f6", "f7", "f8"}
    return agent, estimator, pad


def test_live_loop_pause_during_inference_prevents_dispatch(monkeypatch):
    def pause_on_first_decision(count, callbacks):
        if count == 1:
            callbacks["f8"]()

    agent, _, pad = run_live_loop(monkeypatch, [(1, 1, True)] * 6, pause_on_first_decision)
    assert agent.calls > 0
    assert pad.presses == 0


def test_live_loop_never_infers_on_stale_frames_and_latches_pause(monkeypatch, capsys):
    samples = [(1, 1, True)] * 4 + [(1, 1, False)] * 7 + [(1, 1, True)] * 4
    agent, _, pad = run_live_loop(monkeypatch, samples)
    assert "reason=stale_capture" in capsys.readouterr().out
    assert agent.calls == 4  # two before the outage, two after history warmup
    assert pad.presses == 2  # one held guard before the outage, no automatic resume


def test_live_loop_resets_policy_and_perception_between_rounds(monkeypatch):
    samples = [(1, 1, True)] * 3 + [(0.5, 0.4, True), (0.5, 0, True), (0.5, 0, True)]
    samples += [(1, 1, True)] * 4
    agent, estimator, _ = run_live_loop(monkeypatch, samples)
    assert agent.resets == estimator.resets == 2
    assert agent.calls == 4

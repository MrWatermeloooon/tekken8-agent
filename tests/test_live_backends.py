from __future__ import annotations

import numpy as np
import threading
import time

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.io.screen_backend import DxcamScreenStateBackend
from t8_agent.moves.notation import parse_command
from t8_agent.sim.actions import SimAction


class FakeGamepad:
    def __init__(self) -> None:
        self.pressed: list[str] = []
        self.released: list[str] = []
        self.updates = 0
        self.held: set[str] = set()

    def press_button(self, *, button: str) -> None:
        self.pressed.append(button)
        self.held.add(button)

    def release_button(self, *, button: str) -> None:
        self.released.append(button)
        self.held.discard(button)

    def update(self) -> None:
        self.updates += 1


class FakeCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.stopped = False

    def grab(self) -> np.ndarray:
        return self.frame

    def stop(self) -> None:
        self.stopped = True


def test_virtual_controller_maps_sim_actions_to_buttons() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=gamepad, facing=1, tap_seconds=0.0)

    backend.send_action(SimAction.THROW)

    assert "x" in gamepad.pressed
    assert "a" in gamepad.pressed
    assert gamepad.updates >= 2


def test_virtual_controller_respects_facing_for_forward() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=gamepad, facing=-1, tap_seconds=0.0)

    backend.send_action(SimAction.WALK_FORWARD)

    assert "left" in gamepad.pressed


def test_virtual_controller_maps_expanded_actions() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=gamepad, facing=1, tap_seconds=0.0)

    backend.send_action(SimAction.LOW_PARRY)
    backend.send_action(SimAction.THROW_BREAK_1_2)
    backend.send_action(SimAction.HEAT_BURST)

    assert "down" in gamepad.pressed
    assert "right" in gamepad.pressed
    assert "x" in gamepad.pressed
    assert "y" in gamepad.pressed
    assert "a" in gamepad.pressed


def test_virtual_controller_can_remap_attack_buttons() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(
        gamepad=gamepad,
        facing=1,
        tap_seconds=0.0,
        lp_button="a",
        rp_button="b",
        lk_button="x",
        rk_button="y",
    )

    backend.send_action(SimAction.THROW)
    backend.send_action(SimAction.HOPKICK)

    assert "a" in gamepad.pressed
    assert "x" in gamepad.pressed
    assert "y" in gamepad.pressed


def test_virtual_controller_double_taps_dash_actions() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=gamepad, facing=1, tap_seconds=0.0, dash_gap_seconds=0.0)

    backend.send_action(SimAction.DASH_FORWARD)

    assert gamepad.pressed.count("right") == 2


def test_catalog_command_executor_handles_motion_chords_and_facing() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(
        gamepad=gamepad, facing=-1, tap_seconds=0.0, dash_gap_seconds=0.0,
        frame_seconds=0.0,
    )

    backend.send_command(parse_command("qcf+1+2"), move_id="test:1")

    assert "down" in gamepad.pressed
    assert "left" in gamepad.pressed
    assert "x" in gamepad.pressed and "y" in gamepad.pressed
    assert not gamepad.held


def test_catalog_command_executor_prepares_while_standing() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(
        gamepad=gamepad, tap_seconds=0.0, dash_gap_seconds=0.0,
        charge_seconds=0.0, frame_seconds=0.0,
    )

    backend.send_command(parse_command("ws1"), move_id="jun:97")

    assert gamepad.pressed[:2] == ["down", "x"]
    assert not gamepad.held


def test_catalog_command_executor_rejects_unvalidated_notation() -> None:
    backend = VGamepadInputBackend(gamepad=FakeGamepad(), tap_seconds=0.0)
    with np.testing.assert_raises_regex(ValueError, "no validated executable command"):
        backend.send_command(parse_command("Back throw"), move_id="jun:140")


def test_virtual_controller_can_flip_facing_after_side_switch() -> None:
    gamepad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=gamepad, facing=1, tap_seconds=0.0)

    assert backend.flip_facing() == -1
    backend.send_action(SimAction.WALK_FORWARD)

    assert "left" in gamepad.pressed


def test_screen_backend_reads_calibrated_health_regions(tmp_path) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    frame[0:10, 0:10] = [180, 40, 40]
    frame[0:10, 20:30] = [180, 40, 40]
    frame[0:10, 25:30] = [0, 0, 0]
    config = tmp_path / "screen.yaml"
    config.write_text(
        "p1_health_region: [0, 0, 10, 10]\n"
        "p2_health_region: [20, 0, 30, 10]\n",
        encoding="utf-8",
    )
    camera = FakeCamera(frame)
    backend = DxcamScreenStateBackend(config_path=config, camera=camera)

    state = backend.read()
    backend.close()

    assert state.p1.health == 180.0
    assert state.p2.health == 90.0
    assert state.raw is not None
    assert state.raw["has_health_calibration"] is True
    assert camera.stopped is True


def test_screen_backend_estimates_fighter_positions_from_body_regions(tmp_path) -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[25:35, 12:18] = [40, 180, 40]
    frame[25:35, 60:66] = [40, 40, 180]
    config = tmp_path / "screen.yaml"
    config.write_text(
        "p1_body_region: [0, 20, 40, 40]\n"
        "p2_body_region: [40, 20, 80, 40]\n"
        "position_distance_scale: 0.25\n",
        encoding="utf-8",
    )
    backend = DxcamScreenStateBackend(config_path=config, camera=FakeCamera(frame))

    state = backend.read()

    assert state.raw is not None
    assert state.raw["has_position_calibration"] is True
    assert state.raw["p1_x"] < state.raw["p2_x"]
    assert state.p1.position_x == state.raw["p1_x"]
    assert state.p2.position_x == state.raw["p2_x"]
    assert state.distance < 1.2
    assert state.raw["position_distance_scale"] == 0.25


def test_screen_backend_applies_two_point_distance_calibration(tmp_path) -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[25:35, 12:18] = [40, 180, 40]
    frame[25:35, 60:66] = [40, 40, 180]
    config = tmp_path / "screen.yaml"
    config.write_text(
        "p1_body_region: [0, 20, 40, 40]\n"
        "p2_body_region: [40, 20, 80, 40]\n"
        "position_distance_near_raw: 4.0\n"
        "position_distance_near_sim: 0.7\n"
        "position_distance_far_raw: 4.5\n"
        "position_distance_far_sim: 3.0\n",
        encoding="utf-8",
    )
    backend = DxcamScreenStateBackend(config_path=config, camera=FakeCamera(frame))

    state = backend.read()

    assert state.raw is not None
    assert state.raw["position_distance_calibrated"] is True
    assert 0.44 <= state.distance <= 7.2


def test_screen_backend_swaps_player_positions_after_side_switch(tmp_path) -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[25:35, 12:18] = [40, 180, 40]
    frame[25:35, 60:66] = [40, 40, 180]
    config = tmp_path / "screen.yaml"
    config.write_text(
        "p1_body_region: [0, 20, 40, 40]\n"
        "p2_body_region: [40, 20, 80, 40]\n",
        encoding="utf-8",
    )
    backend = DxcamScreenStateBackend(config_path=config, camera=FakeCamera(frame))

    left_state = backend.read()
    backend.set_p1_on_left(False)
    right_state = backend.read()

    assert left_state.p1.position_x < left_state.p2.position_x
    assert right_state.p1.position_x > right_state.p2.position_x
    assert right_state.p1.facing == -1
    assert right_state.raw is not None and right_state.raw["p1_on_left"] is False


def test_guard_persists_without_release_between_decisions() -> None:
    pad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=pad, tap_seconds=0.0)
    backend.send_action(SimAction.BLOCK_LOW)
    assert pad.held == {"down", "left"}
    releases = len(pad.released)
    backend.send_action(SimAction.BLOCK_LOW)
    assert len(pad.released) == releases
    backend.send_action(SimAction.WALK_BACK)
    assert pad.held == {"left"}
    backend.release_all()
    assert not pad.held


def test_pause_cancels_remaining_dash_and_rejects_stale_decision() -> None:
    released_first_tap = threading.Event()

    class SignalingPad(FakeGamepad):
        def release_button(self, *, button):
            super().release_button(button=button)
            released_first_tap.set()

    pad = SignalingPad()
    backend = VGamepadInputBackend(gamepad=pad, tap_seconds=0.005, dash_gap_seconds=1.0)
    old_generation = backend.generation
    thread = threading.Thread(target=backend.send_action, args=(SimAction.DASH_FORWARD,))
    thread.start()
    try:
        assert released_first_tap.wait(1.0)
        backend.set_enabled(False)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert pad.pressed == ["right"]
        assert not pad.held
        assert not backend.send_action(SimAction.JAB)
        backend.set_enabled(True)
        assert not backend.send_action(SimAction.JAB, expected_generation=old_generation)
    finally:
        backend.close()
        thread.join(timeout=1.0)


def test_asynchronous_hold_watchdog_disables_and_releases() -> None:
    pad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=pad, asynchronous=True, hold_timeout=0.05)
    try:
        assert backend.send_action(SimAction.BLOCK_LOW)
        deadline = time.perf_counter() + 1.0
        while backend.enabled and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert not backend.enabled
        assert not pad.held
        assert not backend.send_action(SimAction.JAB)
    finally:
        backend.close()


def test_jump_step_and_walk_have_distinct_timing_and_facing() -> None:
    pad = FakeGamepad()
    backend = VGamepadInputBackend(gamepad=pad, sidestep_seconds=0.0, dash_gap_seconds=0.0)
    assert backend._timed_sequence("jump")[0][1] > 0.0
    backend.send_action(SimAction.SIDESTEP_LEFT)
    assert pad.pressed == ["up"] and not pad.held
    pad.pressed.clear()
    backend.send_action(SimAction.SIDEWALK_LEFT)
    assert pad.pressed == ["up", "up"] and pad.held == {"up"}
    backend.send_action(SimAction.SIDEWALK_LEFT)
    assert pad.pressed == ["up", "up"]
    backend.flip_facing()
    backend.send_action(SimAction.SIDEWALK_LEFT)
    assert pad.held == {"down"}
    backend.close()


def test_capture_loss_marks_cached_frame_invalid_and_preserves_age(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr("t8_agent.io.screen_backend.perf_counter", lambda: clock[0])
    camera = FakeCamera(np.zeros((20, 40, 3), dtype=np.uint8))
    backend = DxcamScreenStateBackend(camera=camera)
    first = backend.read()
    assert first.raw["capture_fresh"]
    camera.frame = None
    clock[0] += 1.0
    stale = backend.read()
    assert not stale.raw["capture_valid"]
    assert not stale.raw["capture_fresh"]
    assert stale.raw["frame_age_seconds"] == 1.0
    assert stale.raw["frame_id"] == first.raw["frame_id"]


def test_capture_owns_frame_instead_of_aliasing_camera_buffer() -> None:
    camera = FakeCamera(np.zeros((20, 40, 3), dtype=np.uint8))
    backend = DxcamScreenStateBackend(camera=camera)
    backend.read()
    camera.frame[:] = 255
    assert not backend.last_frame.any()

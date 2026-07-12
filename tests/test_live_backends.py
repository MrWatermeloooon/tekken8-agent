from __future__ import annotations

import os

import numpy as np

from t8_agent.io.controller_backend import VGamepadInputBackend
from t8_agent.io.screen_backend import DxcamScreenStateBackend
from t8_agent.live.agents import find_latest_checkpoint
from t8_agent.sim.tekken_lite import SimAction


class FakeGamepad:
    def __init__(self) -> None:
        self.pressed: list[str] = []
        self.released: list[str] = []
        self.updates = 0

    def press_button(self, *, button: str) -> None:
        self.pressed.append(button)

    def release_button(self, *, button: str) -> None:
        self.released.append(button)

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


def test_find_latest_checkpoint_uses_checkpoint_tree(tmp_path) -> None:
    older = tmp_path / "old.zip"
    newer = tmp_path / "nested" / "new.zip"
    newer.parent.mkdir()
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    os.utime(older, (100.0, 100.0))
    os.utime(newer, (200.0, 200.0))

    assert find_latest_checkpoint(tmp_path) == newer

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

    assert "lp" in gamepad.pressed
    assert "lk" in gamepad.pressed
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
    assert "lp" in gamepad.pressed
    assert "rp" in gamepad.pressed
    assert "lk" in gamepad.pressed


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


def test_find_latest_checkpoint_uses_checkpoint_tree(tmp_path) -> None:
    older = tmp_path / "old.zip"
    newer = tmp_path / "nested" / "new.zip"
    newer.parent.mkdir()
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    os.utime(older, (100.0, 100.0))
    os.utime(newer, (200.0, 200.0))

    assert find_latest_checkpoint(tmp_path) == newer

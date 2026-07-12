from __future__ import annotations

import time
from typing import Any

from t8_agent.core.types import DiscreteAction
from t8_agent.io.input_backend import InputBackend
from t8_agent.sim.tekken_lite import SimAction


ActionLike = DiscreteAction | SimAction | str


class VGamepadInputBackend(InputBackend):
    """Virtual Xbox controller backend for local/offline game testing."""

    def __init__(
        self,
        *,
        facing: int = 1,
        tap_seconds: float = 0.055,
        dash_gap_seconds: float = 0.035,
        lp_button: str = "x",
        rp_button: str = "y",
        lk_button: str = "a",
        rk_button: str = "b",
        gamepad: Any | None = None,
    ) -> None:
        self.facing = 1 if facing >= 0 else -1
        self.tap_seconds = tap_seconds
        self.dash_gap_seconds = dash_gap_seconds
        self.layout = {
            "lp": lp_button,
            "rp": rp_button,
            "lk": lk_button,
            "rk": rk_button,
        }
        self._vgamepad = None
        if gamepad is None:
            try:
                import vgamepad as vgamepad_module
            except ImportError as exc:
                raise RuntimeError(
                    "vgamepad is not installed. Install live extras with: "
                    '.\\.venv\\Scripts\\python -m pip install -e ".[live]"'
                ) from exc
            self._vgamepad = vgamepad_module
            gamepad = vgamepad_module.VX360Gamepad()
        self.gamepad = gamepad
        self._buttons = self._resolve_buttons()

    def send(self, action: DiscreteAction) -> None:
        self.send_action(action)

    def flip_facing(self) -> int:
        """Reverse directional inputs after the fighters switch sides."""
        self.release_all()
        self.facing *= -1
        return self.facing

    def send_action(self, action: ActionLike) -> None:
        action_value = action.value if hasattr(action, "value") else str(action)
        self.release_all()
        sequence = self._button_sequence_for_action(action_value)
        for idx, buttons in enumerate(sequence):
            for button in buttons:
                self.gamepad.press_button(button=button)
            self.gamepad.update()
            if buttons:
                time.sleep(self.tap_seconds)
            self.release_all()
            if idx < len(sequence) - 1:
                time.sleep(self.dash_gap_seconds)

    def release_all(self) -> None:
        for button in self._buttons.values():
            self.gamepad.release_button(button=button)
        self.gamepad.update()

    def _resolve_buttons(self) -> dict[str, Any]:
        if self._vgamepad is not None:
            button_cls = self._vgamepad.XUSB_BUTTON
            return {
                "up": button_cls.XUSB_GAMEPAD_DPAD_UP,
                "down": button_cls.XUSB_GAMEPAD_DPAD_DOWN,
                "left": button_cls.XUSB_GAMEPAD_DPAD_LEFT,
                "right": button_cls.XUSB_GAMEPAD_DPAD_RIGHT,
                "x": button_cls.XUSB_GAMEPAD_X,
                "y": button_cls.XUSB_GAMEPAD_Y,
                "a": button_cls.XUSB_GAMEPAD_A,
                "b": button_cls.XUSB_GAMEPAD_B,
            }
        return {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "x": "x",
            "y": "y",
            "a": "a",
            "b": "b",
        }

    def _logical_button(self, name: str) -> Any:
        return self._buttons[self.layout[name]]

    def _button_sequence_for_action(self, action_value: str) -> list[list[Any]]:
        forward = "right" if self.facing == 1 else "left"
        back = "left" if self.facing == 1 else "right"
        if action_value == SimAction.DASH_FORWARD.value:
            return [[self._buttons[forward]], [self._buttons[forward]]]
        if action_value == SimAction.DASH_BACK.value:
            return [[self._buttons[back]], [self._buttons[back]]]
        return [self._buttons_for_action(action_value)]

    def _buttons_for_action(self, action_value: str) -> list[Any]:
        forward = "right" if self.facing == 1 else "left"
        back = "left" if self.facing == 1 else "right"
        mapping = {
            DiscreteAction.NEUTRAL.value: [],
            DiscreteAction.WALK_FORWARD.value: [forward],
            DiscreteAction.WALK_BACK.value: [back],
            DiscreteAction.CROUCH.value: ["down"],
            DiscreteAction.JUMP.value: ["up"],
            DiscreteAction.LEFT_PUNCH.value: ["lp"],
            DiscreteAction.RIGHT_PUNCH.value: ["rp"],
            DiscreteAction.LEFT_KICK.value: ["lk"],
            DiscreteAction.RIGHT_KICK.value: ["rk"],
            SimAction.NEUTRAL.value: [],
            SimAction.WALK_FORWARD.value: [forward],
            SimAction.WALK_BACK.value: [back],
            SimAction.DASH_FORWARD.value: [forward],
            SimAction.DASH_BACK.value: [back],
            SimAction.CROUCH.value: ["down"],
            SimAction.STAND.value: [],
            SimAction.JUMP.value: ["up"],
            SimAction.SIDESTEP_LEFT.value: ["up"],
            SimAction.SIDESTEP_RIGHT.value: ["down"],
            SimAction.SIDEWALK_LEFT.value: ["up"],
            SimAction.SIDEWALK_RIGHT.value: ["down"],
            SimAction.BLOCK_HIGH.value: [back],
            SimAction.BLOCK_LOW.value: ["down", back],
            SimAction.LOW_PARRY.value: ["down", forward],
            SimAction.THROW_BREAK_1.value: ["lp"],
            SimAction.THROW_BREAK_2.value: ["rp"],
            SimAction.THROW_BREAK_1_2.value: ["lp", "rp"],
            SimAction.JAB.value: ["lp"],
            SimAction.DF1.value: ["down", forward, "lp"],
            SimAction.F2.value: [forward, "rp"],
            SimAction.DB3.value: ["down", back, "lk"],
            SimAction.HOPKICK.value: ["up", forward, "rk"],
            SimAction.THROW.value: ["lp", "lk"],
            SimAction.HEAT_BURST.value: ["rp", "lk"],
            SimAction.HEAT_SMASH.value: ["rp", "lk"],
            SimAction.RAGE_ART.value: ["down", forward, "lp", "rp"],
        }
        buttons = []
        for name in mapping.get(action_value, []):
            if name in self.layout:
                buttons.append(self._logical_button(name))
            else:
                buttons.append(self._buttons[name])
        return buttons

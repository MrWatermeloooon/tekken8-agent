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
        gamepad: Any | None = None,
    ) -> None:
        self.facing = 1 if facing >= 0 else -1
        self.tap_seconds = tap_seconds
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

    def send_action(self, action: ActionLike) -> None:
        action_value = action.value if hasattr(action, "value") else str(action)
        self.release_all()
        buttons = self._buttons_for_action(action_value)
        for button in buttons:
            self.gamepad.press_button(button=button)
        self.gamepad.update()
        if buttons:
            time.sleep(self.tap_seconds)
            self.release_all()

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
                "lp": button_cls.XUSB_GAMEPAD_X,
                "rp": button_cls.XUSB_GAMEPAD_Y,
                "lk": button_cls.XUSB_GAMEPAD_A,
                "rk": button_cls.XUSB_GAMEPAD_B,
            }
        return {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "lp": "lp",
            "rp": "rp",
            "lk": "lk",
            "rk": "rk",
        }

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
            SimAction.BLOCK_HIGH.value: [back],
            SimAction.BLOCK_LOW.value: ["down", back],
            SimAction.JAB.value: ["lp"],
            SimAction.DF1.value: ["down", forward, "lp"],
            SimAction.F2.value: [forward, "rp"],
            SimAction.DB3.value: ["down", back, "lk"],
            SimAction.HOPKICK.value: ["up", forward, "rk"],
            SimAction.THROW.value: ["lp", "lk"],
        }
        return [self._buttons[name] for name in mapping.get(action_value, [])]

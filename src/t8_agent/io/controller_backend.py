from __future__ import annotations

import time
import math
import threading
from collections import deque
from typing import Any

from t8_agent.core.types import DiscreteAction
from t8_agent.io.input_backend import InputBackend
from t8_agent.moves.notation import CommandSpec, InputStep
from t8_agent.sim.actions import SimAction


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
        asynchronous: bool = False,
        hold_timeout: float = 0.25,
        sidestep_seconds: float = 0.025,
        jump_seconds: float = 0.15,
        charge_seconds: float = 0.25,
        frame_seconds: float = 1.0 / 60.0,
    ) -> None:
        self.facing = 1 if facing >= 0 else -1
        self.tap_seconds = tap_seconds
        self.dash_gap_seconds = dash_gap_seconds
        if any(not math.isfinite(value) or value < 0 for value in
               (tap_seconds, dash_gap_seconds, sidestep_seconds, jump_seconds,
                charge_seconds, frame_seconds)):
            raise ValueError("input durations must be finite and nonnegative")
        if not math.isfinite(hold_timeout) or hold_timeout <= 0:
            raise ValueError("hold_timeout must be positive and finite")
        self.sidestep_seconds = sidestep_seconds
        self.jump_seconds = jump_seconds
        self.charge_seconds = charge_seconds
        self.frame_seconds = frame_seconds
        self.hold_timeout = hold_timeout
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
        self._condition = threading.Condition(threading.RLock())
        self._held: set[Any] = set()
        self._stages: deque[tuple[set[Any], float | None]] = deque()
        self._deadline: float | None = None
        self._active_action: str | None = None
        self._enabled = True
        self._closed = False
        self._generation = 0
        self._last_heartbeat = time.perf_counter()
        self._worker_error: Exception | None = None
        self._worker = None
        if asynchronous:
            self._worker = threading.Thread(target=self._run, name="tekken-controller", daemon=True)
            self._worker.start()

    def send(self, action: DiscreteAction) -> None:
        self.send_action(action)

    def flip_facing(self) -> int:
        """Reverse directional inputs after the fighters switch sides."""
        with self._condition:
            self._cancel_locked()
            self.facing *= -1
            return self.facing

    def send_action(self, action: ActionLike, *, expected_generation: int | None = None) -> bool:
        action_value = action.value if hasattr(action, "value") else str(action)
        with self._condition:
            if self._worker_error is not None:
                raise RuntimeError("controller worker failed") from self._worker_error
            if not self._enabled or self._closed or (
                expected_generation is not None and expected_generation != self._generation
            ):
                return False
            now = time.perf_counter()
            self._advance_locked(now)
            if self._deadline is not None:
                return False
            self._last_heartbeat = now
            if action_value == self._active_action and self._held:
                return True
            self._cancel_locked()
            self._active_action = action_value
            self._stages.extend(self._timed_sequence(action_value))
            self._advance_locked(now)
            generation = self._generation
            self._condition.notify_all()
            # Calibration callers can wait; live play uses the independent worker.
            while self._worker is None and self._deadline is not None:
                self._condition.wait(max(0.0, self._deadline - time.perf_counter()))
                if generation != self._generation:
                    return False
                self._advance_locked(time.perf_counter())
            return True

    def send_command(
        self,
        command: CommandSpec | dict[str, Any],
        *,
        move_id: str,
        expected_generation: int | None = None,
    ) -> bool:
        """Dispatch one validated catalog command as a cancellable input sequence."""

        spec = self._coerce_command(command)
        if not spec.executable:
            raise ValueError(f"move {move_id} has no validated executable command")
        unsupported = set(spec.requirements) - {"H", "R", "WS", "FC", "SS", "BT", "WR", "HFC"}
        if unsupported:
            raise ValueError(
                f"move {move_id} requires unresolved state(s): {', '.join(sorted(unsupported))}"
            )
        stages = self._command_sequence(spec, move_id)
        with self._condition:
            if self._worker_error is not None:
                raise RuntimeError("controller worker failed") from self._worker_error
            if not self._enabled or self._closed or (
                expected_generation is not None and expected_generation != self._generation
            ):
                return False
            now = time.perf_counter()
            self._advance_locked(now)
            if self._deadline is not None:
                return False
            self._last_heartbeat = now
            self._cancel_locked()
            self._active_action = move_id
            self._stages.extend(stages)
            self._advance_locked(now)
            generation = self._generation
            self._condition.notify_all()
            while self._worker is None and self._deadline is not None:
                self._condition.wait(max(0.0, self._deadline - time.perf_counter()))
                if generation != self._generation:
                    return False
                self._advance_locked(time.perf_counter())
            return True

    def release_all(self) -> None:
        with self._condition:
            self._cancel_locked()

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @property
    def is_busy(self) -> bool:
        with self._condition:
            return self._deadline is not None

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled and not self._closed

    def set_enabled(self, enabled: bool) -> None:
        with self._condition:
            self._cancel_locked()
            self._enabled = bool(enabled) and not self._closed
            self._last_heartbeat = time.perf_counter()

    def heartbeat(self) -> None:
        with self._condition:
            self._last_heartbeat = time.perf_counter()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._enabled = False
            self._cancel_locked()
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    def _set_buttons_locked(self, buttons: set[Any]) -> None:
        for button in self._held - buttons:
            self.gamepad.release_button(button=button)
        for button in buttons - self._held:
            self.gamepad.press_button(button=button)
        if buttons != self._held:
            self.gamepad.update()
        self._held = buttons

    def _cancel_locked(self) -> None:
        self._generation += 1
        self._stages.clear()
        self._deadline = None
        self._active_action = None
        self._set_buttons_locked(set())
        self._condition.notify_all()

    def _advance_locked(self, now: float) -> None:
        if self._deadline is not None and now < self._deadline:
            return
        self._deadline = None
        while self._stages:
            buttons, duration = self._stages.popleft()
            self._set_buttons_locked(buttons)
            if duration is None:
                break
            if duration > 0:
                # Never compress a missed pulse because the worker woke up late.
                self._deadline = now + duration
                break

    def _run(self) -> None:
        with self._condition:
            try:
                while not self._closed:
                    now = time.perf_counter()
                    if self._enabled and now - self._last_heartbeat >= self.hold_timeout:
                        self._enabled = False
                        self._cancel_locked()
                    self._advance_locked(now)
                    wait = 0.01 if self._deadline is None else min(0.01, max(0.0, self._deadline - now))
                    self._condition.wait(wait)
            except Exception as exc:
                self._worker_error = exc
                self._enabled = False
                self._cancel_locked()

    def _timed_sequence(self, action: str) -> list[tuple[set[Any], float | None]]:
        buttons = set(self._buttons_for_action(action))
        if action in {"neutral", "stand", "walk_forward", "walk_back", "crouch", "block_high", "block_low"}:
            return [(buttons, None)]
        if action in {"sidewalk_left", "sidewalk_right"}:
            return [(buttons, self.sidestep_seconds), (set(), self.dash_gap_seconds), (buttons, None)]
        if action in {"dash_forward", "dash_back"}:
            return [(buttons, self.tap_seconds), (set(), self.dash_gap_seconds),
                    (buttons, self.tap_seconds), (set(), None)]
        duration = (self.jump_seconds if action == "jump" else
                    self.sidestep_seconds if action.startswith("sidestep_") else self.tap_seconds)
        return [(buttons, duration), (set(), None)]

    @staticmethod
    def _coerce_command(command: CommandSpec | dict[str, Any]) -> CommandSpec:
        if isinstance(command, CommandSpec):
            return command
        steps = tuple(InputStep(
            raw=str(step["raw"]),
            direction=str(step.get("direction", "n")),
            buttons=tuple(int(value) for value in step.get("buttons", ())),
            hold=bool(step.get("hold", False)),
            just_frame=bool(step.get("just_frame", False)),
            delay_min=int(step.get("delay_min", 0)),
            delay_max=int(step.get("delay_max", 0)),
            release=bool(step.get("release", False)),
            requirement=step.get("requirement"),
            uncertain=bool(step.get("uncertain", False)),
        ) for step in command.get("steps", ()))
        return CommandSpec(
            raw=str(command.get("raw", "")),
            requirements=tuple(str(value) for value in command.get("requirements", ())),
            steps=steps,
            parser_status=str(command.get("parser_status", "invalid")),
            warnings=tuple(str(value) for value in command.get("warnings", ())),
        )

    def _command_sequence(
        self,
        command: CommandSpec,
        move_id: str,
    ) -> list[tuple[set[Any], float | None]]:
        stages: list[tuple[set[Any], float | None]] = []
        for step in command.steps:
            if step.uncertain:
                raise ValueError(f"move {move_id} contains an unresolved input: {step.raw}")
            requirement = (step.requirement or "").upper()
            if requirement == "WS":
                stages.extend([(self._direction_buttons("d"), self.charge_seconds),
                               (set(), self.frame_seconds)])
            elif requirement == "SS":
                stages.extend([(self._direction_buttons("u"), self.sidestep_seconds),
                               (set(), self.frame_seconds)])
            elif requirement in {"BT", "CH"}:
                raise ValueError(f"move {move_id} requires observed {requirement} state")

            motion = self._motion_directions(step.direction)
            for direction in motion[:-1]:
                stages.append((self._direction_buttons(direction), self.frame_seconds))
            final_direction = motion[-1] if motion else "n"
            buttons = self._direction_buttons(final_direction)
            buttons.update(self._attack_buttons(step.buttons))
            if step.release:
                buttons = set()
            duration = self.charge_seconds if step.hold and not step.buttons else self.tap_seconds
            if step.delay_min > 0:
                stages.append((set(), step.delay_min * self.frame_seconds))
            stages.append((buttons, duration))
            stages.append((set(), self.frame_seconds if step.just_frame else self.dash_gap_seconds))
        stages.append((set(), None))
        return stages

    def _motion_directions(self, direction: str) -> list[str]:
        normalized = direction.lower()
        motions = {
            "qcf": ["d", "df", "f"],
            "qcb": ["d", "db", "b"],
            "hcf": ["b", "db", "d", "df", "f"],
            "hcb": ["f", "df", "d", "db", "b"],
        }
        return motions.get(normalized, [normalized])

    def _direction_buttons(self, direction: str) -> set[Any]:
        forward = "right" if self.facing == 1 else "left"
        back = "left" if self.facing == 1 else "right"
        mapping = {
            "n": (), "f": (forward,), "b": (back,), "u": ("up",), "d": ("down",),
            "uf": ("up", forward), "ub": ("up", back),
            "df": ("down", forward), "db": ("down", back),
        }
        if direction not in mapping:
            raise ValueError(f"unsupported controller direction: {direction}")
        return {self._buttons[value] for value in mapping[direction]}

    def _attack_buttons(self, buttons: tuple[int, ...]) -> set[Any]:
        names = {1: "lp", 2: "rp", 3: "lk", 4: "rk"}
        return {self._logical_button(names[value]) for value in buttons}

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
            SimAction.SIDESTEP_LEFT.value: ["up" if self.facing == 1 else "down"],
            SimAction.SIDESTEP_RIGHT.value: ["down" if self.facing == 1 else "up"],
            SimAction.SIDEWALK_LEFT.value: ["up" if self.facing == 1 else "down"],
            SimAction.SIDEWALK_RIGHT.value: ["down" if self.facing == 1 else "up"],
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

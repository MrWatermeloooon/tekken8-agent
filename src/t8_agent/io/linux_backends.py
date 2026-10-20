"""Linux screen capture and virtual controller for the live runtime.

The Windows runtime uses DXcam (Desktop Duplication) and vgamepad (ViGEmBus).
On Linux the same interfaces are provided by:

* ``MssCamera``: ``mss`` screen grabs (X11; on Wayland run the game under
  XWayland). ``grab()`` returns an RGB frame or ``None`` when called faster
  than the capture rate, matching DXcam's "no new frame" contract. mss cannot
  tell a repeated frame from a new one, so stalls are detected by the
  runtime's frame-age check only when grabs fail.
* ``UinputGamepad``: an evdev ``UInput`` device that presents itself as an
  Xbox 360 controller (D-pad as hat axes, face buttons A/B/X/Y), so Steam
  Input and Proton map it like a physical pad. Writing to /dev/uinput needs
  permission (the ``input`` group or a udev rule).

Neither has been validated against Tekken 8 under Proton yet; check the
capture regions and the controller mapping in Practice mode first, exactly as
on Windows. Install with ``pip install -e ".[live-linux]"``.
"""
from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np


class MssCamera:
    def __init__(self, *, output_idx: int = 0, fps: int = 60, grabber: Any | None = None) -> None:
        if grabber is None:
            try:
                import mss
            except ImportError as exc:
                raise RuntimeError('mss is not installed; install with: pip install -e ".[live-linux]"') from exc
            grabber = mss.mss()
        self._grabber = grabber
        monitors = grabber.monitors
        # monitors[0] is the union of all displays; [1:] are the individual ones.
        self._monitor = monitors[output_idx + 1] if len(monitors) > output_idx + 1 else monitors[0]
        self._interval = 1.0 / max(1, fps)
        self._last = 0.0

    def grab(self) -> np.ndarray | None:
        now = time.perf_counter()
        if now - self._last < self._interval:
            return None
        self._last = now
        shot = self._grabber.grab(self._monitor)
        frame = np.asarray(shot, dtype=np.uint8)  # BGRA
        return np.ascontiguousarray(frame[:, :, 2::-1])  # RGB, like DXcam's default

    def stop(self) -> None:
        close = getattr(self._grabber, "close", None)
        if close is not None:
            close()


class UinputGamepad:
    """Xbox 360-style virtual pad; button names match VGamepadInputBackend's logical names."""

    VENDOR, PRODUCT = 0x045E, 0x028E

    def __init__(self, *, device: Any | None = None, ecodes: Any | None = None) -> None:
        if device is None or ecodes is None:
            try:
                from evdev import AbsInfo, UInput, ecodes as evdev_codes
            except ImportError as exc:
                raise RuntimeError('evdev is not installed; install with: pip install -e ".[live-linux]"') from exc
            ecodes = evdev_codes
            capabilities = {
                ecodes.EV_KEY: [ecodes.BTN_A, ecodes.BTN_B, ecodes.BTN_X, ecodes.BTN_Y,
                                ecodes.BTN_TL, ecodes.BTN_TR, ecodes.BTN_START, ecodes.BTN_SELECT],
                ecodes.EV_ABS: [(ecodes.ABS_HAT0X, AbsInfo(0, -1, 1, 0, 0, 0)),
                                (ecodes.ABS_HAT0Y, AbsInfo(0, -1, 1, 0, 0, 0))],
            }
            device = UInput(capabilities, name="Microsoft X-Box 360 pad", vendor=self.VENDOR,
                            product=self.PRODUCT, version=0x110)
        self._device = device
        self._codes = ecodes
        self._keys = {"a": ecodes.BTN_A, "b": ecodes.BTN_B, "x": ecodes.BTN_X, "y": ecodes.BTN_Y}
        self._held: set[str] = set()
        self._sent_keys: dict[str, bool] = {}
        self._sent_hat = (0, 0)

    def press_button(self, *, button: str) -> None:
        self._held.add(str(button))

    def release_button(self, *, button: str) -> None:
        self._held.discard(str(button))

    def update(self) -> None:
        codes = self._codes
        for name, code in self._keys.items():
            pressed = name in self._held
            if self._sent_keys.get(name, False) != pressed:
                self._device.write(codes.EV_KEY, code, 1 if pressed else 0)
                self._sent_keys[name] = pressed
        # Opposite directions cancel, as on a physical D-pad.
        hat_x = (1 if "right" in self._held else 0) - (1 if "left" in self._held else 0)
        hat_y = (1 if "down" in self._held else 0) - (1 if "up" in self._held else 0)
        if (hat_x, hat_y) != self._sent_hat:
            if hat_x != self._sent_hat[0]:
                self._device.write(codes.EV_ABS, codes.ABS_HAT0X, hat_x)
            if hat_y != self._sent_hat[1]:
                self._device.write(codes.EV_ABS, codes.ABS_HAT0Y, hat_y)
            self._sent_hat = (hat_x, hat_y)
        self._device.syn()

    def close(self) -> None:
        close = getattr(self._device, "close", None)
        if close is not None:
            close()


def default_camera(output_idx: int = 0, fps: int = 60) -> Any:
    """DXcam on Windows, mss elsewhere."""
    if sys.platform == "win32":
        import dxcam
        return dxcam.create(output_idx=output_idx)
    return MssCamera(output_idx=output_idx, fps=fps)


def default_gamepad() -> Any | None:
    """None on Windows (VGamepadInputBackend creates a ViGEm pad), uinput elsewhere."""
    return None if sys.platform == "win32" else UinputGamepad()

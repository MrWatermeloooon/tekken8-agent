"""Practice-mode menu automation with the virtual pad: character selection by reading names.

The character select grid (Tekken 8, 2026-10-11) has three rows; the cursor skips locked slots
and wraps around within a row, so presses are not counted: after every move the hovered name is
read (the big name on the P1 / P2 side, matched against templates in
data/character_select_names.npz learned from the grid) until it is the target. Used by
scripts/collect_characters.py for the overnight data collection.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np

TEMPLATES = Path(__file__).resolve().parents[3] / "data" / "character_select_names.npz"
# The band holding the hovered character's big name (full resolution, 1920x1080), and where the
# name is anchored: P1's starts at x=90 (left-aligned), P2's ends at x=1830 (right-aligned).
NAME_BAND = (540, 660)
NAME_ANCHORS = {1: ("left", 90), 2: ("right", 1830)}
NAME_HEIGHT = 32  # letters are scaled to this height for matching


def _letters(crop_bgr: np.ndarray, side: str, anchor: float) -> tuple[np.ndarray, str | None] | None:
    """The name as a binary image at NAME_HEIGHT, and which side of it ("left"/"right") is cut
    off, if any.

    Name letters are bright, low-saturation blobs 30-75 px tall lying inside the band; the
    fighter's model behind (a grey suit, a patterned jacket) also has bright pixels, so only
    letter-sized blobs chained to the anchor edge (gaps up to 40 px, a space included) count.
    Where the model touches a letter the two merge into one big blob that is not a letter: the
    name is then cut there (and so it is at the crop's edge)."""
    pixels = crop_bgr.astype(int)
    bright = ((pixels.min(axis=2) > 150) & (pixels.max(axis=2) - pixels.min(axis=2) < 70)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    height = crop_bgr.shape[0]
    every = [tuple(int(v) for v in stats[i, :4]) for i in range(1, count) if stats[i, 4] >= 60]
    blobs = [b for b in every if 30 <= b[3] <= 75 and b[1] > 0 and b[1] + b[3] < height]
    if not blobs:
        return None
    if side == "left":
        start = min((b for b in blobs if abs(b[0] - anchor) < 60), key=lambda b: b[0], default=None)
    else:
        start = max((b for b in blobs if abs(b[0] + b[2] - anchor) < 60), key=lambda b: b[0] + b[2], default=None)
    if start is None:
        return None
    chain = [start]
    top, bottom = start[1], start[1] + start[3]

    def gap_to(b, left, right):
        return b[0] - right if side == "left" else left - (b[0] + b[2])

    while True:
        left, right = min(b[0] for b in chain), max(b[0] + b[2] for b in chain)
        nxt = [b for b in blobs if b not in chain and -5 <= gap_to(b, left, right) <= 40
               and abs(b[1] - top) < 12 and abs(b[1] + b[3] - bottom) < 12]
        if not nxt:
            break
        chain.append(min(nxt, key=lambda b: b[0]) if side == "left" else max(nxt, key=lambda b: b[0] + b[2]))
    left, right = min(b[0] for b in chain), max(b[0] + b[2] for b in chain)
    top, bottom = min(b[1] for b in chain), max(b[1] + b[3] for b in chain)
    far = "right" if side == "left" else "left"
    at_edge = right >= crop_bgr.shape[1] - 1 if side == "left" else left <= 1
    # A non-letter blob overlapping the name's rows right where the chain stopped.
    # (Its box may start well before the stop: a body part behind the letters.)
    lo, hi = (right, right + 40) if side == "left" else (left - 40, left)
    blocked = any(b not in chain and b[1] < bottom and b[1] + b[3] > top and b[0] < hi and b[0] + b[2] > lo
                  for b in every)
    box = bright[top:bottom, left:right].astype(np.float32)
    width = max(4, int(round(box.shape[1] * NAME_HEIGHT / box.shape[0])))
    image = cv2.GaussianBlur(cv2.resize(box, (width, NAME_HEIGHT), interpolation=cv2.INTER_AREA), (3, 3), 0)
    return image, (far if at_edge or blocked else None)


def _distance(query: np.ndarray, query_cut: str | None, template: np.ndarray, template_cut: str | None) -> float:
    """Mean difference over the overlap, aligned on the side neither image is cut on, plus the
    width mismatch. A cut image only has to be no wider than the other, and a cut query must
    show at least 70% of the name (a short prefix such as "LE" fits several names)."""
    if query_cut and template_cut and query_cut != template_cut:
        return 1.0
    width = min(query.shape[1], template.shape[1])
    cut = query_cut or template_cut
    if cut == "right":
        a, b = query[:, :width], template[:, :width]
    else:
        a, b = query[:, -width:], template[:, -width:]
    shape = float(np.abs(a - b).mean())
    wq, wt = query.shape[1], template.shape[1]
    if query_cut and not template_cut:
        mismatch = max(0.0, wq - wt) / wt + max(0.0, 0.7 - wq / wt)
    elif template_cut and not query_cut:
        mismatch = max(0.0, wt - wq) / wt
    elif query_cut:
        mismatch = 0.0
    else:
        mismatch = abs(wq - wt) / wt
    return shape + mismatch


def load_templates(path: Path = TEMPLATES) -> dict:
    """{name: (letter image, cut side or None)}; the last row of each stored array flags the cut."""
    data = np.load(path)
    return {key.replace("_", "-"): (data[key][:-1], {1.0: "left", 2.0: "right"}.get(float(data[key][-1, 0])))
            for key in data.files}


def read_name(frame_rgb: np.ndarray, side: int, templates: dict) -> tuple[str | None, float]:
    scale = frame_rgb.shape[1] / 1920.0
    if abs(scale - 1.0) > 1e-3:
        frame_rgb = cv2.resize(frame_rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    top, bottom = NAME_BAND
    anchor_side, anchor = NAME_ANCHORS[side]
    found = _letters(frame_rgb[top:bottom, :, ::-1], anchor_side, anchor)
    if found is None:
        return None, 1.0
    query, query_cut = found
    scores = {name: _distance(query, query_cut, image, cut) for name, (image, cut) in templates.items()}
    best = min(scores, key=scores.get)
    return best, scores[best]


def build_templates(crops: dict[str, np.ndarray], path: Path = TEMPLATES) -> None:
    """Saves name templates from P2-side crops (BGR, full resolution, band rows NAME_BAND,
    columns starting at `x0`): {name: (crop, x0)}."""
    out = {}
    for name, (crop, x0) in crops.items():
        found = _letters(crop, "right", NAME_ANCHORS[2][1] - x0)
        if found is None:
            raise ValueError(f"no name found in the crop for {name}")
        image, cut = found
        flag = np.zeros((1, image.shape[1]), np.float32)
        flag[0, 0] = {"left": 1.0, "right": 2.0}.get(cut, 0.0)
        out[name.replace("-", "_")] = np.vstack([image, flag])
    np.savez_compressed(path, **out)


class CharacterSelect:
    def __init__(self, camera: Any, press: Callable[[str], None], grab: Callable[[], np.ndarray | None]) -> None:
        """press(name) taps a pad button ("up", "down", "left", "right", "a", "b", "start");
        grab() returns the newest RGB frame."""
        data = np.load(TEMPLATES)
        # Stored per name: the letter image and whether it was cut by the crop's edge.
        self.templates = load_templates()
        self.press = press
        self.grab = grab

    def hovered(self, side: int, frames: int = 3) -> tuple[str | None, float]:
        """The hovered character on a side, and the match distance (lower is surer): the best of a
        few frames, as the fighter model behind may hide a letter in one."""
        best: tuple[str | None, float] = (None, 1.0)
        for _ in range(frames):
            frame = self.grab()
            if frame is not None:
                name, distance = read_name(frame, side, self.templates)
                if distance < best[1]:
                    best = (name, distance)
                if distance < 0.05:
                    break
        return best

    def select(self, target: str, side: int, confirm: bool = True) -> bool:
        """Moves the side's cursor to `target` by reading names, and confirms with A.

        A first walk over the grid wants a sure read (distance < 0.15). A name whose template
        reads less surely on this side (one learned from an old crop) can miss that; if the
        walk saw the target at all, a second walk accepts the best distance it saw."""
        best = 1.0
        for attempt in range(2):
            threshold = 0.15 if attempt == 0 else min(0.35, best + 0.05)
            for _ in range(3):
                self.press("up")
            for _ in range(3):  # rows
                for _ in range(17):
                    time.sleep(0.15)
                    name, distance = self.hovered(side)
                    if name == target:
                        best = min(best, distance)
                        if distance < threshold:
                            if confirm:
                                self.press("a")
                            return True
                    self.press("right")
                self.press("down")
            if best >= 0.35:
                return False  # never even close: not on the grid
        return False


def pad_presser(controller: Any) -> Callable[[str], None]:
    """A press(name) for the virtual Xbox pad of a VGamepadInputBackend."""
    buttons = controller._vgamepad.XUSB_BUTTON  # noqa: SLF001 (setup-only raw access)
    mapping = {"up": buttons.XUSB_GAMEPAD_DPAD_UP, "down": buttons.XUSB_GAMEPAD_DPAD_DOWN,
               "left": buttons.XUSB_GAMEPAD_DPAD_LEFT, "right": buttons.XUSB_GAMEPAD_DPAD_RIGHT,
               "a": buttons.XUSB_GAMEPAD_A, "b": buttons.XUSB_GAMEPAD_B, "start": buttons.XUSB_GAMEPAD_START}
    gamepad = controller.gamepad

    def press(name: str, hold: float = 0.08, after: float = 0.25) -> None:
        controller.release_all()
        gamepad.press_button(button=mapping[name])
        gamepad.update()
        time.sleep(hold)
        gamepad.release_button(button=mapping[name])
        gamepad.update()
        time.sleep(after)

    return press

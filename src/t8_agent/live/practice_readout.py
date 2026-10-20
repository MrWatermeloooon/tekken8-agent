"""Reads Practice mode's "Distance from Opponent" number off the screen.

The Practice info panel (bottom left; enable it in Practice's display settings) shows the exact
in-game distance in a clean fixed font. Digits are matched against templates learned from
labelled crops (``build_templates``), so no OCR package is needed. Used by the automated
distance calibration (scripts/auto_calibrate_distance.py).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Where the panel is, as fractions of the full frame (left, top, right, bottom), and where the
# number sits inside that panel crop: the right end of its last line.
PANEL = (0.03, 0.58, 0.28, 0.92)
NUMBER = (0.80, 0.86, 1.0, 0.97)
NUMBER_ROWS = 18  # the digits fill the top of the number box; below is stage floor
TEMPLATE_SIZE = (12, 16)
DEFAULT_TEMPLATES = Path(__file__).resolve().parents[3] / "data" / "practice_digits.npz"
# Digit templates for read_line_value (its lower threshold draws thicker glyphs).
LINE_TEMPLATES = Path(__file__).resolve().parents[3] / "data" / "practice_line_digits.npz"


def panel_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    left, top, right, bottom = PANEL
    return frame[int(height * top):int(height * bottom), int(width * left):int(width * right)]


def _glyphs(panel: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """The number's characters, left to right, as (binary glyph image, its ink height)."""
    height, width = panel.shape[:2]
    left, top, right, bottom = NUMBER
    box = panel[int(height * top):int(height * bottom), int(width * left):int(width * right)]
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY) if box.ndim == 3 else box
    gray = gray[:NUMBER_ROWS]
    binary = (gray > 150).astype(np.uint8)
    inked = binary.sum(axis=0) > 0
    glyphs, start = [], None
    for column, ink in enumerate(list(inked) + [False]):
        if ink and start is None:
            start = column
        elif not ink and start is not None:
            piece = binary[:, start:column]
            rows = np.flatnonzero(piece.sum(axis=1))
            glyphs.append((piece[rows.min():rows.max() + 1], int(rows.max() - rows.min() + 1)))
            start = None
    return glyphs


def _normalize(glyph: np.ndarray) -> np.ndarray:
    return cv2.resize(glyph.astype(np.float32), TEMPLATE_SIZE, interpolation=cv2.INTER_AREA).ravel()


def build_templates(panels: list[np.ndarray], values: list[str], path: Path = DEFAULT_TEMPLATES) -> dict[str, int]:
    """Learns digit templates from panel crops whose numbers are known (e.g. "1.94")."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for panel, value in zip(panels, values):
        digits = value.replace(".", "")
        found = _glyphs(panel)
        glyphs = [g for g, h in found if h >= 0.7 * max(hh for _, hh in found)][:len(digits)] if found else []
        if len(glyphs) != len(digits):
            continue  # segmentation disagrees with the label: skip this crop
        for glyph, digit in zip(glyphs, digits):
            sums[digit] = sums.get(digit, 0) + _normalize(glyph)
            counts[digit] = counts.get(digit, 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"d{d}": sums[d] / counts[d] for d in sums})
    return counts


class PracticeDistanceReader:
    def __init__(self, path: Path = DEFAULT_TEMPLATES) -> None:
        data = np.load(path)
        self.templates = {key[1:]: data[key] for key in data.files}

    def read_panel(self, panel: np.ndarray) -> float | None:
        """The distance shown, or None if the number cannot be read.

        The readout is always one digit, a point, and two decimals (d.dd); the point is too
        faint to segment reliably, so the format places it, and only the first three digit
        glyphs count (a speck of stage behind the panel can add a spurious one)."""
        glyphs = _glyphs(panel)
        if not glyphs:
            return None
        tallest = max(height for _, height in glyphs)
        digits = [glyph for glyph, height in glyphs if height >= 0.7 * tallest]
        if len(digits) < 3:
            return None
        text = ""
        for glyph in digits[:3]:
            vector = _normalize(glyph)
            text += min(self.templates, key=lambda digit: float(np.abs(self.templates[digit] - vector).sum()))
        return int(text[0]) + int(text[1:]) / 100.0

    def read_frame(self, frame: np.ndarray) -> float | None:
        return self.read_panel(panel_crop(frame))


# Panel lines (vertical centre as a fraction of the panel crop's height) whose value is read.
LINES = {
    "total_damage": 0.045,
    "damage": 0.327,
    "startup": 0.630,
    "frame_advantage": 0.714,
    "distance": 0.890,
}


def _line_glyphs(panel: np.ndarray, centre: float, half_height: int = 11) -> list[list[tuple[np.ndarray, int, int]]]:
    """The value column of one panel line, as words of glyphs (image, ink height, left column).
    The brightest channel is used, so coloured values (frame advantage) read like white ones."""
    height, width = panel.shape[:2]
    row = int(height * centre)
    band = panel[max(0, row - half_height):row + half_height, int(width * 0.55):]
    bright = band.max(axis=2) if band.ndim == 3 else band
    # 110, not 150: coloured values (a cyan or red frame advantage) peak around 170 in their
    # brightest channel, and their thin strokes drop out at 150; the panel background is ~40.
    binary = (bright > 110).astype(np.uint8)
    inked = binary.sum(axis=0) > 0
    glyphs, start = [], None
    for column, ink in enumerate(list(inked) + [False]):
        if ink and start is None:
            start = column
        elif not ink and start is not None:
            piece = binary[:, start:column]
            rows = np.flatnonzero(piece.sum(axis=1))
            glyphs.append((piece[rows.min():rows.max() + 1], int(rows.max() - rows.min() + 1), start, column))
            start = None
    words: list[list[tuple[np.ndarray, int, int]]] = []
    for glyph, glyph_height, left, right in glyphs:
        if words and left - words[-1][-1][3] <= 6:
            words[-1].append((glyph, glyph_height, left, right))
        else:
            words.append([(glyph, glyph_height, left, right)])
    return words


def build_line_templates(panels: list[np.ndarray], distances: list[str], path: Path = LINE_TEMPLATES) -> dict[str, int]:
    """Learns read_line_value's digit templates from panels whose distance line is known."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for panel, value in zip(panels, distances):
        words = _line_glyphs(panel, LINES["distance"])
        if not words:
            continue
        tallest = max(height for _, height, _, _ in words[0])
        glyphs = [glyph for glyph, height, _, _ in words[0] if height >= 0.6 * tallest]
        digits = value.replace(".", "")
        if len(glyphs) < len(digits):
            continue
        for glyph, digit in zip(glyphs, digits):
            sums[digit] = sums.get(digit, 0) + _normalize(glyph)
            counts[digit] = counts.get(digit, 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"d{d}": sums[d] / counts[d] for d in sums})
    return counts


class PracticeLineReader:
    """Reads values off any panel line (read_line_value) with its own digit templates."""

    def __init__(self, path: Path = LINE_TEMPLATES) -> None:
        data = np.load(path)
        self.templates = {key[1:]: data[key] for key in data.files}

    def read(self, panel: np.ndarray, line: str) -> int | float | None:
        return read_line_value(self, panel, line)

    def read_frame(self, frame: np.ndarray, line: str) -> int | float | None:
        return read_line_value(self, panel_crop(frame), line)


def read_line_value(reader, panel: np.ndarray, line: str) -> int | float | None:
    """The first value on a panel line: startup frames, damage, a signed frame advantage, or the
    distance. None for "---" (nothing measured yet) or anything unreadable.

    The value is read up to its opening bracket ("10 (10 - 10)", "+8 (---)", "5 (100%)"): a
    bracket is taller and thinner than a digit. A plus is a small square cross, a minus a flat
    dash, the decimal point a small dot."""
    words = _line_glyphs(panel, LINES[line])
    if not words:
        return None
    glyphs = words[0]  # the value is the first word; anything after it is a bracket or stray stage
    tallest = max(height for _, height, _, _ in glyphs)
    if tallest < 6:
        return None
    # Digits are the shortest of the tall glyphs (brackets are a little taller still).
    digit_height = min(height for _, height, _, _ in glyphs if height >= 0.8 * tallest)
    sign, digits, dashes = 1, [], 0
    for glyph, height, _, _ in glyphs:
        rows, cols = glyph.shape
        if height > 1.08 * digit_height and cols <= 0.45 * height:
            break  # the opening bracket ends the value
        if rows <= 3 and cols >= 4:
            dashes += 1  # a minus, or part of "---"
            continue
        if not digits and 0.3 * digit_height <= height <= 0.8 * digit_height and _is_plus(glyph):
            continue  # a plus sign
        if height < 0.6 * digit_height:
            continue  # the decimal point
        if dashes == 1 and not digits:
            sign = -1
        vector = _normalize(glyph)
        digits.append(min(reader.templates, key=lambda digit: float(np.abs(reader.templates[digit] - vector).sum())))
    if not digits:
        return None  # "---": nothing measured
    if line == "distance":
        return int(digits[0]) + int("".join(digits[1:3]) or 0) / 100.0
    return sign * int("".join(digits))


def _is_plus(glyph: np.ndarray) -> bool:
    """A plus: ink along the middle row and the middle column, little in the corners."""
    rows, cols = glyph.shape
    if rows < 5 or cols < 5:
        return False
    middle_row = glyph[rows // 2].mean()
    middle_col = glyph[:, cols // 2].mean()
    corners = (glyph[:rows // 4, :cols // 4].mean() + glyph[-rows // 4:, -cols // 4:].mean()) / 2
    return middle_row > 0.6 and middle_col > 0.6 and corners < 0.2

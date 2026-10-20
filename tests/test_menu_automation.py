"""Character-select and menu helpers of the overnight collector, on synthetic frames."""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_characters import dialog_highlight, settled_reading, stage_hovered  # noqa: E402
from t8_agent.live.menu_automation import NAME_BAND, _distance, _letters, read_name  # noqa: E402


def name_frame(text: str, side: int, body: bool = False) -> np.ndarray:
    """An RGB 1080p frame with a white name where the select screen draws it (and, optionally,
    a bright body part touching its last letter, as Victor's lapel does)."""
    frame = np.full((1080, 1920, 3), 30, np.uint8)
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
    x = 90 if side == 1 else 1830 - width
    cv2.putText(frame, text, (x, 625), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (235, 235, 235), 5)
    if body:
        cv2.rectangle(frame, (x + width - 12, 560), (x + width + 60, 700), (200, 200, 200), -1)
    return frame


def templates_from(names: list[str]) -> dict:
    out = {}
    for name in names:
        top, bottom = NAME_BAND
        image, cut = _letters(name_frame(name.upper(), 2)[top:bottom, :, ::-1], "right", 1830)
        out[name] = (image, cut)
    return out


def test_names_read_on_both_sides():
    templates = templates_from(["victor", "jin", "kazuya", "lee", "leo"])
    for name in templates:
        assert read_name(name_frame(name.upper(), 1), 1, templates)[0] == name
        assert read_name(name_frame(name.upper(), 2), 2, templates)[0] == name


def test_body_touching_the_last_letter_cuts_the_name_but_still_reads():
    templates = templates_from(["victor", "jin", "kazuya", "reina", "clive"])
    top, bottom = NAME_BAND
    image, cut = _letters(name_frame("VICTOR", 1, body=True)[top:bottom, :, ::-1], "left", 90)
    assert cut == "right"
    name, distance = read_name(name_frame("VICTOR", 1, body=True), 1, templates)
    assert name == "victor" and distance < 0.15


def test_short_prefix_does_not_match():
    wide = np.ones((32, 100), np.float32)
    assert _distance(wide[:, :30], "right", wide, None) > 0.3  # 30% of a name is not enough
    assert _distance(wide[:, :80], "right", wide, None) < 0.05


def test_settled_reading_keeps_the_last_startup_before_the_panel_clears():
    hit = {"damage": 5, "startup": 10}
    cleared = {"damage": 5, "startup": None}
    assert settled_reading([{}, hit, hit, hit, cleared, cleared]) == hit
    assert settled_reading([cleared, cleared]) == cleared
    assert settled_reading([hit]) == {}


def test_dialog_and_stage_detectors():
    frame = np.full((1080, 1920, 3), 30, np.uint8)
    assert dialog_highlight(frame) is None and stage_hovered(frame) is None
    no = frame.copy()
    no[598:638, 544:1376] = (170, 35, 64)  # the red bar on "No"
    assert dialog_highlight(no) == "no"
    yes = frame.copy()
    yes[552:592, 544:1376] = (170, 35, 64)
    assert dialog_highlight(yes) == "yes"
    arena = frame.copy()
    arena[704:800, 430:580] = (220, 60, 120)  # Arena's tile glows pink when hovered
    assert stage_hovered(arena) == "arena"
    random_tile = frame.copy()
    cv2.rectangle(random_tile, (255, 705), (405, 799), (220, 40, 70), 6)  # just a red frame
    assert stage_hovered(random_tile) == "random"

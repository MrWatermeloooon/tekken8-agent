"""The pose tracker picks the two fighters and reads where they stand (no model download needed)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from t8_agent.io.pose_tracker import LEFT_ANKLE, LEFT_HIP, RIGHT_ANKLE, RIGHT_HIP, PoseFighterTracker


def person(x1, y1, x2, y2, confidence, ankle_x, hip_y, ankle_y):
    points = np.zeros((17, 3), np.float32)
    points[[LEFT_ANKLE, RIGHT_ANKLE]] = [[ankle_x - 10, ankle_y, 0.9], [ankle_x + 10, ankle_y, 0.9]]
    points[[LEFT_HIP, RIGHT_HIP]] = [[ankle_x - 10, hip_y, 0.9], [ankle_x + 10, hip_y, 0.9]]
    return (x1, y1, x2, y2), confidence, points


class FakeModel:
    def __init__(self, people):
        self.people = people

    def predict(self, frame, **kwargs):
        boxes = SimpleNamespace(xyxy=torch.tensor([p[0] for p in self.people], dtype=torch.float32),
                                conf=torch.tensor([p[1] for p in self.people]))
        keypoints = SimpleNamespace(data=torch.tensor(np.stack([p[2] for p in self.people])))
        result = SimpleNamespace(boxes=_Sized(boxes, len(self.people)), keypoints=keypoints)
        return [result]


class _Sized:
    def __init__(self, inner, count):
        self.inner, self.count = inner, count

    def __len__(self):
        return self.count

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_picks_the_two_largest_people_left_to_right_by_their_feet():
    frame = np.zeros((1000, 2000, 3), np.uint8)
    llama = person(1700, 500, 1900, 800, 0.9, 1800, 650, 790)       # smaller figure in the background
    right = person(1100, 100, 1400, 900, 0.8, 1250, 500, 880)
    left = person(300, 120, 600, 900, 0.6, 470, 520, 880)
    tracker = PoseFighterTracker({"pose_band_top": 0.0}, model=FakeModel([llama, right, left]))
    pair = tracker.update(frame)
    assert pair is not None
    assert [round(f.x) for f in pair] == [470, 1250], "ankle midpoints, ordered left to right"
    assert all(f.posture == "standing" for f in pair)


def test_keeps_the_last_pair_when_a_fighter_is_missed_and_reads_crouching():
    frame = np.zeros((1000, 2000, 3), np.uint8)
    standing = [person(300, 100, 600, 900, 0.8, 450, 500, 880), person(1100, 100, 1400, 900, 0.8, 1250, 500, 880)]
    tracker = PoseFighterTracker({"pose_band_top": 0.0}, model=FakeModel(standing))
    first = tracker.update(frame)
    assert tracker.fresh
    tracker.model = FakeModel(standing[:1])
    assert tracker.update(frame) == first, "one person found: keep the last pair"
    assert not tracker.fresh, "a repeated pair is marked stale"
    crouch = [person(300, 400, 600, 900, 0.8, 450, 700, 880), standing[1]]  # hips much lower
    tracker.model = FakeModel(crouch)
    assert tracker.update(frame)[0].posture == "crouching"


def test_background_people_are_not_fighters():
    """At long range a fighter can be missed while ruins or a statue register as small people
    higher up the screen: they must not be taken as the second fighter."""
    frame = np.zeros((1000, 2000, 3), np.uint8)
    fighter = person(1700, 300, 1850, 750, 0.9, 1780, 560, 740)
    ruins = person(900, 200, 960, 340, 0.4, 930, 280, 335)   # small, standing high up
    tracker = PoseFighterTracker({"pose_band_top": 0.0}, model=FakeModel([fighter, ruins]))
    assert tracker.update(frame) is None and not tracker.fresh
    other = person(80, 320, 240, 745, 0.5, 160, 580, 740)     # the real second fighter, same floor
    tracker.model = FakeModel([fighter, ruins, other])
    pair = tracker.update(frame)
    assert tracker.fresh and [round(f.x) for f in pair] == [160, 1780]

"""Fighter positions and postures from a pretrained pose model (YOLOv8 pose).

The old position reader averaged every colourful pixel in each half of the screen, so it
mostly measured the stage background (positions stuck near the stage edges). This finds the
two fighters as people with skeletons:

* the fighters are the most prominent person in the fighting band (below the HUD) and the
  largest other person at least 60% as tall standing on the same floor line; background
  "people" (statues, ruins, stage animals) are smaller or stand higher up the screen;
* a fighter's x is the midpoint of its ankles (where it stands), else its box centre;
* posture comes from the skeleton: crouching (hips low relative to the standing height seen
  this round), airborne (feet clearly above the floor line), grounded (the body lies flat).

Screen-config keys: ``position_source: pose`` turns it on; ``pose_model`` (default
models/yolov8n-pose.pt), ``pose_device`` (default cpu: training saturates the GPU, and the
nano model runs in ~28 ms on the CPU at 640 px), ``pose_imgsz`` (640: at long range the camera
zooms out and 416 misses a small fighter at the screen edge), ``pose_band_top``
(fraction of the screen height above which is HUD, 0.14), ``pose_min_confidence`` (0.35).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# COCO keypoints used here.
LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 11, 12, 13, 14, 15, 16
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 5, 6
KEYPOINT_CONFIDENCE = 0.4


@dataclass(frozen=True)
class FighterPose:
    x: float              # screen x of where it stands (pixels)
    box: tuple[float, float, float, float]
    confidence: float
    posture: str          # standing, crouching, airborne, grounded, unknown
    keypoints: np.ndarray  # (17, 3): x, y, confidence in screen pixels


class PoseFighterTracker:
    def __init__(self, config: dict[str, Any], model: Any | None = None) -> None:
        self.device = str(config.get("pose_device", "cpu"))
        self.imgsz = int(config.get("pose_imgsz", 640))
        self.band_top = float(config.get("pose_band_top", 0.14))
        self.min_confidence = float(config.get("pose_min_confidence", 0.35))
        if model is None:
            from ultralytics import YOLO, settings
            settings.update({"sync": False})  # no usage analytics
            model = YOLO(str(Path(config.get("pose_model", "models/yolov8n-pose.pt"))))
        self.model = model
        self.floor_y: float | None = None        # the ankles' line while standing, this round
        self.standing_height: float | None = None  # hip-to-ankle height while standing
        self.last: tuple[FighterPose, FighterPose] | None = None
        self.fresh = False  # whether update() found the pair in this frame (else it repeats the last)

    def reset_round(self) -> None:
        self.floor_y = self.standing_height = None
        self.last = None

    def update(self, frame: np.ndarray) -> tuple[FighterPose, FighterPose] | None:
        """The (left, right) fighters on screen, or the last seen pair if this frame has fewer
        than two (None until a pair has been seen)."""
        top = int(frame.shape[0] * self.band_top)
        band = frame[top:]
        result = self.model.predict(band, device=self.device, verbose=False, imgsz=self.imgsz)[0]
        people = []
        boxes = result.boxes
        keypoints = result.keypoints
        for index in range(len(boxes)):
            confidence = float(boxes.conf[index])
            if confidence < self.min_confidence:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[index].tolist())
            points = keypoints.data[index].cpu().numpy().astype(np.float64).copy()
            points[:, 1] += top
            people.append(((x2 - x1) * (y2 - y1), confidence, (x1, y1 + top, x2, y2 + top), points))
        chosen = _fighters(people)
        self.fresh = chosen is not None
        if chosen is None:
            return self.last
        pair = []
        for _, confidence, box, points in chosen:
            pair.append(FighterPose(x=_stance_x(box, points), box=box, confidence=confidence,
                                    posture="unknown", keypoints=points))
        pair.sort(key=lambda fighter: fighter.x)
        self._learn_floor(pair)
        left, right = (FighterPose(f.x, f.box, f.confidence, self._posture(f), f.keypoints) for f in pair)
        self.last = (left, right)
        return self.last

    def _learn_floor(self, pair: list[FighterPose]) -> None:
        """Track the floor line and the standing hip height from upright skeletons."""
        for fighter in pair:
            ankles = _points(fighter.keypoints, (LEFT_ANKLE, RIGHT_ANKLE))
            hips = _points(fighter.keypoints, (LEFT_HIP, RIGHT_HIP))
            if ankles is None or hips is None:
                continue
            height = ankles[:, 1].mean() - hips[:, 1].mean()
            if height <= 0:
                continue
            # The tallest hip-to-ankle span seen is standing; the floor is where standing ankles are.
            if self.standing_height is None or height > self.standing_height * 0.97:
                self.standing_height = height if self.standing_height is None else max(self.standing_height, height)
                floor = float(ankles[:, 1].mean())
                self.floor_y = floor if self.floor_y is None else 0.9 * self.floor_y + 0.1 * floor

    def _posture(self, fighter: FighterPose) -> str:
        """Airborne when the body is clearly off the floor (a launched fighter, juggled: its box
        bottom well above the floor line); grounded when lying (box much wider than tall) on the
        floor; crouching when the hips are low over the feet; else standing. Measured in Practice
        (scripts/check_pose_postures.py): standing and crouching read 100%; a launch reads
        standing, airborne, grounded, getting up, standing. A standing jump barely leaves the
        floor and its tucked legs look like a crouch: it reads crouching."""
        ankles = _points(fighter.keypoints, (LEFT_ANKLE, RIGHT_ANKLE))
        hips = _points(fighter.keypoints, (LEFT_HIP, RIGHT_HIP))
        shoulders = _points(fighter.keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER))
        if self.standing_height is None or self.floor_y is None:
            return "unknown"
        x1, y1, x2, y2 = fighter.box
        if self.floor_y - y2 > 0.15 * self.standing_height:
            return "airborne"
        if (x2 - x1) > 1.3 * (y2 - y1):
            return "grounded"  # lying: much wider than tall, on the floor
        if hips is None:
            return "unknown"
        if ankles is not None:
            height = ankles[:, 1].mean() - hips[:, 1].mean()
            if height < 0.7 * self.standing_height:
                return "crouching"
        if shoulders is not None and shoulders[:, 1].mean() > hips[:, 1].mean():
            return "grounded"
        return "standing"


def _fighters(people: list) -> list | None:
    """The two fighters among (area, confidence, box, keypoints) detections: the most prominent
    person (area x confidence), and the largest other one at least 60% as tall whose feet are on
    the same floor line (box bottoms within 12% of the taller one's height); failing that (a
    juggled fighter is in the air), the largest other one with at least 35% of its area."""
    if len(people) < 2:
        return None
    ranked = sorted(people, key=lambda person: person[0] * person[1], reverse=True)
    first = ranked[0]
    height = first[2][3] - first[2][1]
    floor = first[2][3]
    candidates = [person for person in ranked[1:]
                  if person[2][3] - person[2][1] >= 0.6 * height and abs(person[2][3] - floor) <= 0.12 * height]
    if not candidates:
        # Nobody on the floor line: a launched (juggled) or falling fighter is off it. Take a large
        # detection anywhere (background figures are far smaller than a fighter).
        area = first[0]
        candidates = [person for person in ranked[1:] if person[0] >= 0.35 * area]
    if not candidates:
        return None
    return [first, max(candidates, key=lambda person: person[0])]


def _points(keypoints: np.ndarray, indices: tuple[int, ...]) -> np.ndarray | None:
    chosen = keypoints[list(indices)]
    chosen = chosen[chosen[:, 2] >= KEYPOINT_CONFIDENCE]
    return chosen[:, :2] if len(chosen) else None


def _stance_x(box: tuple[float, float, float, float], keypoints: np.ndarray) -> float:
    ankles = _points(keypoints, (LEFT_ANKLE, RIGHT_ANKLE))
    if ankles is not None:
        return float(ankles[:, 0].mean())
    return (box[0] + box[2]) / 2.0

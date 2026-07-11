from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
import yaml

from t8_agent.core.types import GameState, PlayerState
from t8_agent.io.state_backend import StateBackend


@dataclass(frozen=True)
class ScreenRegion:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_config(cls, data: list[int] | tuple[int, int, int, int] | None) -> "ScreenRegion | None":
        if data is None:
            return None
        if len(data) != 4:
            raise ValueError(f"screen region must be [left, top, right, bottom], got {data}")
        left, top, right, bottom = [int(value) for value in data]
        if right <= left or bottom <= top:
            raise ValueError(f"invalid screen region {data}")
        return cls(left=left, top=top, right=right, bottom=bottom)


class DxcamScreenStateBackend(StateBackend):
    """Screen capture backend with optional health-bar crop estimation."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        output_idx: int = 0,
        max_health: float = 180.0,
        stage_half_width: float = 3.6,
        camera: Any | None = None,
    ) -> None:
        self.max_health = max_health
        self.stage_half_width = stage_half_width
        self.config = _load_config(config_path)
        self.p1_health_region = ScreenRegion.from_config(self.config.get("p1_health_region"))
        self.p2_health_region = ScreenRegion.from_config(self.config.get("p2_health_region"))
        self.p1_body_region = ScreenRegion.from_config(self.config.get("p1_body_region"))
        self.p2_body_region = ScreenRegion.from_config(self.config.get("p2_body_region"))
        self.last_frame: np.ndarray | None = None
        if camera is None:
            try:
                import dxcam
            except ImportError as exc:
                raise RuntimeError(
                    "dxcam is not installed. Install live extras with: "
                    '.\\.venv\\Scripts\\python -m pip install -e ".[live]"'
                ) from exc
            camera = dxcam.create(output_idx=output_idx)
        self.camera = camera

    def read(self) -> GameState:
        frame = None
        for _attempt in range(5):
            frame = self.camera.grab()
            if frame is not None:
                break
            sleep(0.05)
        if frame is None:
            frame = self.last_frame
        if frame is None:
            return GameState(
                p1=PlayerState(health=self.max_health, position_x=-self.stage_half_width * 0.28, facing=1),
                p2=PlayerState(health=self.max_health, position_x=self.stage_half_width * 0.28, facing=-1),
                round_timer=60.0,
                raw={
                    "screen_width": 0,
                    "screen_height": 0,
                    "p1_health_ratio": 1.0,
                    "p2_health_ratio": 1.0,
                    "p1_x": float(-self.stage_half_width * 0.28),
                    "p2_x": float(self.stage_half_width * 0.28),
                    "has_health_calibration": bool(self.p1_health_region and self.p2_health_region),
                    "has_position_calibration": bool(self.p1_body_region and self.p2_body_region),
                    "capture_valid": False,
                },
            )
        self.last_frame = frame
        p1_ratio = _estimate_health_ratio(frame, self.p1_health_region)
        p2_ratio = _estimate_health_ratio(frame, self.p2_health_region)
        p1_x, p2_x = _estimate_fighter_positions(
            frame,
            self.stage_half_width,
            p1_region=self.p1_body_region,
            p2_region=self.p2_body_region,
        )
        return GameState(
            p1=PlayerState(health=self.max_health * p1_ratio, position_x=p1_x, facing=1),
            p2=PlayerState(health=self.max_health * p2_ratio, position_x=p2_x, facing=-1),
            round_timer=60.0,
            raw={
                "screen_width": int(frame.shape[1]),
                "screen_height": int(frame.shape[0]),
                "p1_health_ratio": float(p1_ratio),
                "p2_health_ratio": float(p2_ratio),
                "p1_x": float(p1_x),
                "p2_x": float(p2_x),
                "has_health_calibration": bool(self.p1_health_region and self.p2_health_region),
                "has_position_calibration": bool(self.p1_body_region and self.p2_body_region),
                "capture_valid": True,
            },
        )

    def close(self) -> None:
        if hasattr(self.camera, "stop"):
            self.camera.stop()


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"screen config must be a mapping, got {type(data).__name__}")
    return data


def _estimate_health_ratio(frame: np.ndarray, region: ScreenRegion | None) -> float:
    if region is None:
        return 1.0
    crop = frame[region.top : region.bottom, region.left : region.right]
    if crop.size == 0:
        return 1.0
    red = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    blue = crop[:, :, 2].astype(np.int16)
    bright = np.maximum.reduce([red, green, blue])
    saturated = bright - np.minimum.reduce([red, green, blue])
    filled = (bright > 70) & (saturated > 25)
    ratio = float(filled.mean())
    return max(0.0, min(1.0, ratio))


def _estimate_fighter_positions(
    frame: np.ndarray,
    stage_half_width: float,
    *,
    p1_region: ScreenRegion | None,
    p2_region: ScreenRegion | None,
) -> tuple[float, float]:
    height, width = frame.shape[:2]
    p1_default = -stage_half_width * 0.28
    p2_default = stage_half_width * 0.28
    if p1_region is None:
        p1_region = ScreenRegion(0, int(height * 0.35), width // 2, height)
    if p2_region is None:
        p2_region = ScreenRegion(width // 2, int(height * 0.35), width, height)
    p1_x = _estimate_position_x(frame, p1_region, stage_half_width, fallback=p1_default)
    p2_x = _estimate_position_x(frame, p2_region, stage_half_width, fallback=p2_default)
    if p2_x <= p1_x:
        center = (p1_x + p2_x) / 2.0
        p1_x = center - 0.22
        p2_x = center + 0.22
    return p1_x, p2_x


def _estimate_position_x(frame: np.ndarray, region: ScreenRegion, stage_half_width: float, fallback: float) -> float:
    crop = frame[region.top : region.bottom, region.left : region.right]
    if crop.size == 0:
        return fallback
    channels = crop.astype(np.int16)
    bright = channels.max(axis=2)
    dark = channels.min(axis=2)
    saturated = bright - dark
    foreground = (bright > 45) & (saturated > 18)
    if int(foreground.sum()) < 8:
        return fallback
    xs = np.nonzero(foreground)[1] + region.left
    screen_x = float(xs.mean())
    normalized = (screen_x / max(1, frame.shape[1] - 1)) * 2.0 - 1.0
    return max(-stage_half_width, min(stage_half_width, normalized * stage_half_width))

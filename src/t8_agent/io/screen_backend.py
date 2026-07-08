from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        camera: Any | None = None,
    ) -> None:
        self.max_health = max_health
        self.config = _load_config(config_path)
        self.p1_health_region = ScreenRegion.from_config(self.config.get("p1_health_region"))
        self.p2_health_region = ScreenRegion.from_config(self.config.get("p2_health_region"))
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
        frame = self.camera.grab()
        if frame is None:
            frame = self.last_frame
        if frame is None:
            raise RuntimeError("screen capture returned no frame")
        self.last_frame = frame
        p1_ratio = _estimate_health_ratio(frame, self.p1_health_region)
        p2_ratio = _estimate_health_ratio(frame, self.p2_health_region)
        return GameState(
            p1=PlayerState(health=self.max_health * p1_ratio, position_x=-1.0, facing=1),
            p2=PlayerState(health=self.max_health * p2_ratio, position_x=1.0, facing=-1),
            round_timer=60.0,
            raw={
                "screen_width": int(frame.shape[1]),
                "screen_height": int(frame.shape[0]),
                "p1_health_ratio": float(p1_ratio),
                "p2_health_ratio": float(p2_ratio),
                "has_health_calibration": bool(self.p1_health_region and self.p2_health_region),
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

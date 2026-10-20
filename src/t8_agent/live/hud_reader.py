"""Reads Tekken 8's HUD bars: health, recoverable health, and the Heat gauge, for both players.

Measured on the 1920x1080 HUD (2026-10-11, Practice mode; scripts in runs/ captures):

* health bar: yellow is current health, red is recoverable health, grey is lost. Each bar
  empties from its outer end (P1's from the left, P2's from the right). The bar shimmers with
  flame effects, so a column counts as health when any row of the band shows the bar's colour.
* Heat gauge: the blue bar under the health bar, full while Heat is unused, draining while Heat
  is active, empty once spent.

Regions scale with the frame width from the 1920-pixel layout; the screen config can override
them (hud_* keys). These fill the screen-observation features for recoverable health (2, 3),
the fighter's own Heat (26-28), and the opponent's Heat (31).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

# 1920x1080 layout: x spans and row bands.
DEFAULTS = {
    "hud_health_rows": (76, 93),
    "hud_p1_health": (205, 836),
    "hud_p2_health": (1083, 1720),
    "hud_heat_rows": (122, 129),
    "hud_p1_heat": (536, 880),
    "hud_p2_heat": (1039, 1384),
}


@dataclass(frozen=True)
class HudReading:
    p1_health: float
    p1_recoverable: float
    p2_health: float
    p2_recoverable: float
    p1_heat: float  # gauge fill, 0-1
    p2_heat: float

    def heat_state(self, player: int) -> tuple[bool, float, bool]:
        """(in Heat, fraction of Heat left, Heat available) from a gauge: full is available,
        partly full is active, empty is spent."""
        fill = self.p1_heat if player == 1 else self.p2_heat
        available = fill >= 0.9   # a full gauge reads 0.95-1.0 (its ends are a different shade)
        active = 0.05 < fill < 0.9
        return active, fill if active else 0.0, available


class HudReader:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.layout = {key: tuple(config.get(key, value)) for key, value in DEFAULTS.items()}

    def _scaled(self, frame: np.ndarray, key: str) -> tuple[int, int]:
        scale = frame.shape[1] / 1920.0
        start, end = self.layout[key]
        return int(round(start * scale)), int(round(end * scale))

    def _bar(self, frame: np.ndarray, rows_key: str, span_key: str):
        top, bottom = self._scaled(frame, rows_key)  # rows scale with the width too (16:9)
        left, right = self._scaled(frame, span_key)
        return frame[top:bottom, left:right, :3].astype(np.int16)

    def read(self, frame: np.ndarray) -> HudReading:
        """`frame` is RGB."""
        values = {}
        for player in (1, 2):
            band = self._bar(frame, "hud_health_rows", f"hud_p{player}_health")
            r, g, b = band[..., 0], band[..., 1], band[..., 2]
            health = ((r > 220) & (g > 120) & (b < 170)).any(axis=0)
            recoverable = ((r > 160) & (g < 60) & (b < 60)).any(axis=0) & ~health
            width = max(1, band.shape[1])
            values[f"p{player}_health"] = float(health.sum()) / width
            values[f"p{player}_recoverable"] = float(recoverable.sum()) / width
            gauge = self._bar(frame, "hud_heat_rows", f"hud_p{player}_heat")
            gr, gg, gb = gauge[..., 0], gauge[..., 1], gauge[..., 2]
            blue = ((gb > 190) & (gg > 80) & (gr < 140)).mean(axis=0) >= 0.4
            values[f"p{player}_heat"] = float(blue.mean())
        return HudReading(**values)


class HudTracker:
    """HudReader with the median of the last few readings: hit flashes and the Heat activation
    animation make single frames read 0 or half (measured), a median of 5 (~1/3 s) rides that out."""

    def __init__(self, reader: HudReader | None = None, window: int = 5) -> None:
        self.reader = reader or HudReader()
        self.window = window
        self.history: list[HudReading] = []

    def reset_round(self) -> None:
        self.history.clear()

    def update(self, frame: np.ndarray) -> HudReading:
        self.history.append(self.reader.read(frame))
        del self.history[:-self.window]
        fields = HudReading.__dataclass_fields__
        return HudReading(**{name: float(np.median([getattr(r, name) for r in self.history])) for name in fields})

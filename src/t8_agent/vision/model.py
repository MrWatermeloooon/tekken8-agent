from __future__ import annotations

from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
from torch import nn

from t8_agent.vision.temporal import VisualEstimate


VISUAL_TARGET_KEYS = [
    "p1_health_ratio",
    "p2_health_ratio",
    "p1_x",
    "p2_x",
    "distance",
    "p1_velocity",
    "p2_velocity",
    "p1_motion",
    "p2_motion",
    "p1_hit_event",
    "p2_hit_event",
    "p1_attack_likelihood",
    "p2_attack_likelihood",
]


class TemporalStateNet(nn.Module):
    def __init__(self, output_dim: int = len(VISUAL_TARGET_KEYS), hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 6, hidden_dim),
            nn.ReLU(),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = frames.shape
        encoded = self.encoder(frames.reshape(batch * steps, channels, height, width))
        encoded = encoded.reshape(batch, steps, -1)
        sequence, _hidden = self.temporal(encoded)
        return self.head(sequence[:, -1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "target_keys": VISUAL_TARGET_KEYS}, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "TemporalStateNet":
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        model = cls(output_dim=len(payload.get("target_keys", VISUAL_TARGET_KEYS)))
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return model


class LearnedTemporalEstimator:
    def __init__(self, checkpoint: str | Path, *, clip_length: int = 8, device: str = "cpu") -> None:
        self.device = device
        self.model = TemporalStateNet.load(checkpoint, device=device)
        self.frames: deque[np.ndarray] = deque(maxlen=clip_length)

    def update(self, frame: np.ndarray) -> VisualEstimate | None:
        resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        tensor_frame = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        self.frames.append(tensor_frame)
        if len(self.frames) < self.frames.maxlen:
            return None
        inputs = torch.from_numpy(np.stack(self.frames)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            values = self.model(inputs)[0].detach().cpu().numpy()
        return VisualEstimate(
            p1_health_ratio=float(np.clip(values[0], 0.0, 1.0)),
            p2_health_ratio=float(np.clip(values[1], 0.0, 1.0)),
            p1_x=float(values[2]),
            p2_x=float(values[3]),
            distance=max(0.0, float(values[4])),
            p1_velocity=float(values[5]),
            p2_velocity=float(values[6]),
            p1_motion=max(0.0, float(values[7])),
            p2_motion=max(0.0, float(values[8])),
            p1_hit_event=bool(values[9] >= 0.5),
            p2_hit_event=bool(values[10] >= 0.5),
            p1_attack_likelihood=float(np.clip(values[11], 0.0, 1.0)),
            p2_attack_likelihood=float(np.clip(values[12], 0.0, 1.0)),
        )

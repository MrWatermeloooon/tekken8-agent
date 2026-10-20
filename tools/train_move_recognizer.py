"""Trains the opponent-move recognizer on recorded pose sequences (scripts/record_move_dataset.py).

    python tools/train_move_recognizer.py [--data runs/move_dataset] [--epochs 60]

From a fighter's last WINDOW pose frames (normalized to its hips and size, mirrored so it always
faces right, plus joint velocities) it predicts what the screen observation lacks, the
simulator's opponent-move features: is an attack coming (50), at which level (51-54: high, mid,
low, throw), and in how many frames (55). Labels: the move starts at its first clear motion
after the input was sent (so controller and capture latency drop out); a hit lands
`startup` frames later; "attack coming" covers the move's start up to its first hit plus its
active frames. --data is one character's folder or the dataset root (every character folder in
it). About a fifth of the moves (by a fixed hash of the move id, all their repetitions) are held
out for the test, so the reported accuracy is on moves the model never saw. Saves
models/move_recognizer.pt and a report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch
from torch import nn

WINDOW = 8
FPS = 60.0
LEVELS = ["high", "mid", "low", "throw"]
JOINTS = 17


def normalize(keypoints: np.ndarray, face_right: bool) -> np.ndarray:
    """(T, 17, 3) -> (T, 17, 3): x, y relative to the hip centre over the body height, mirrored
    to face right; confidence kept."""
    out = keypoints.copy()
    hips = keypoints[:, [11, 12], :2].mean(axis=1, keepdims=True)
    top = keypoints[:, :, 1].min(axis=1, keepdims=True)
    bottom = keypoints[:, :, 1].max(axis=1, keepdims=True)
    height = np.maximum(bottom - top, 1.0)[:, :, None]
    out[:, :, :2] = (keypoints[:, :, :2] - hips) / height
    if not face_right:
        out[:, :, 0] *= -1.0
    return out


def motion_onset(keypoints: np.ndarray, times: np.ndarray, sent_at: float) -> float | None:
    """The first time after the input was sent when the skeleton moves clearly (a joint shifts
    by more than 4% of the body height between frames), as the move's start."""
    normalized = normalize(keypoints, True)
    for index in range(1, len(times)):
        if times[index] < sent_at:
            continue
        shift = np.abs(normalized[index, :, :2] - normalized[index - 1, :, :2]).max()
        if shift > 0.04:
            return float(times[index - 1])
    return None


def features(sequence: np.ndarray) -> np.ndarray:
    """(WINDOW, 17, 3) normalized -> flat: positions, confidences, and frame-to-frame velocities."""
    velocity = np.diff(sequence[:, :, :2], axis=0, prepend=sequence[:1, :, :2])
    return np.concatenate([sequence.reshape(-1), velocity.reshape(-1)]).astype(np.float32)


def held_out(stable_id: str) -> bool:
    """A fixed fifth of the moves (and of the non-attacks) form the test set."""
    return zlib.crc32(stable_id.encode("utf-8")) % 5 == 0


def load(data: Path):
    rows = {"train": [], "test": []}
    paths = sorted(data.glob("*.npz")) or sorted(data.glob("*/*.npz"))
    for path in paths:
        sample = np.load(path)
        labels = json.loads(str(sample["labels"]))
        times, p1 = sample["times"], sample["p1"]
        if len(times) < WINDOW + 2:
            continue
        # A non-attack is the same for everyone: tell them apart by character for the split.
        split_id = labels["stable_id"] if labels["attack"] else f"{path.parent.name}:{labels['stable_id']}"
        normalized = normalize(p1, True)  # P1 is on the left, facing right
        onset = motion_onset(p1, times, float(sample["sent_at"])) if labels["attack"] else None
        startup = labels.get("measured_startup") or labels.get("startup")
        levels = [level for level in labels.get("hit_levels", []) if level]
        first_level = levels[0] if levels else None
        level_index = {"special_mid": "mid", "special_low": "low"}.get(first_level, first_level)
        for end in range(WINDOW, len(times) + 1):
            t = times[end - 1]
            attacking, level, frames_to_hit = 0.0, -1, 0.0
            if onset is not None and startup:
                elapsed = (t - onset) * FPS
                if 0.0 <= elapsed <= startup + 3:
                    attacking = 1.0
                    level = LEVELS.index(level_index) if level_index in LEVELS else -1
                    frames_to_hit = max(0.0, startup - elapsed)
            rows["test" if held_out(split_id) else "train"].append(
                (features(normalized[end - WINDOW:end]), attacking, level, frames_to_hit))
    return rows


class Recognizer(nn.Module):
    def __init__(self, inputs: int, dropout: float = 0.0, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout))
        self.attack = nn.Linear(128, 1)
        self.level = nn.Linear(128, len(LEVELS))
        self.timing = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(x)
        return self.attack(h).squeeze(-1), self.level(h), self.timing(h).squeeze(-1)


def tensors(rows):
    x = torch.tensor(np.stack([r[0] for r in rows]))
    attack = torch.tensor([r[1] for r in rows])
    level = torch.tensor([r[2] for r in rows])
    timing = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    return x, attack, level, timing


def evaluate(model, x, attack, level, timing) -> dict:
    with torch.no_grad():
        a, l, t = model(x)
    predicted = a > 0
    truth = attack > 0.5
    tp = int((predicted & truth).sum()); fp = int((predicted & ~truth).sum()); fn = int((~predicted & truth).sum())
    mask = truth & (level >= 0)
    level_acc = float((l[mask].argmax(-1) == level[mask]).float().mean()) if mask.any() else float("nan")
    timing_mae = float((t[truth] * 30 - timing[truth]).abs().mean()) if truth.any() else float("nan")
    return {"windows": len(x), "attack_precision": tp / max(tp + fp, 1), "attack_recall": tp / max(tp + fn, 1),
            "level_accuracy": level_acc, "frames_to_hit_mae": timing_mae}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("runs/move_dataset"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("models/move_recognizer.pt"))
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--jitter", type=float, default=0.3, help="input noise in standard deviations (augmentation)")
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()
    rows = load(args.data)
    if not rows["train"] or not rows["test"]:
        raise SystemExit(f"not enough data in {args.data} (need moves on both sides of the split)")
    train, test = tensors(rows["train"]), tensors(rows["test"])
    mean, std = train[0].mean(0), train[0].std(0) + 1e-4
    norm = lambda x: (x - mean) / std  # noqa: E731
    model = Recognizer(train[0].shape[1], args.dropout, args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    positive = float(train[1].mean())
    attack_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - positive) / max(positive, 1e-3)))
    x, attack, level, timing = norm(train[0]), *train[1:]
    for epoch in range(args.epochs):
        order = torch.randperm(len(x))
        for start in range(0, len(x), 256):
            batch = order[start:start + 256]
            model.train()
            a, l, t = model(x[batch] + args.jitter * torch.randn_like(x[batch]))
            loss = attack_loss(a, attack[batch])
            mask = (attack[batch] > 0.5) & (level[batch] >= 0)
            if mask.any():
                loss = loss + nn.functional.cross_entropy(l[mask], level[batch][mask])
                loss = loss + 0.5 * nn.functional.smooth_l1_loss(t[mask], timing[batch][mask] / 30.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    report = {"train": evaluate(model, x, attack, level, timing),
              "test": evaluate(model, norm(test[0]), *test[1:]),
              "positive_share": positive, "window": WINDOW}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "inputs": train[0].shape[1],
                "window": WINDOW, "levels": LEVELS}, args.out)
    Path("runs/move_recognizer_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from t8_agent.vision.model import TemporalStateNet, VISUAL_TARGET_KEYS


class RecordedClipDataset(Dataset):
    def __init__(self, root: str | Path, clip_length: int) -> None:
        self.samples: list[tuple[Path, list[dict]]] = []
        for metadata_path in sorted(Path(root).glob("*/frames.jsonl")):
            records = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            for end in range(clip_length - 1, len(records)):
                clip = records[end - clip_length + 1 : end + 1]
                self.samples.append((metadata_path.parent / "frames", clip))
        if not self.samples:
            raise ValueError(f"no recorded clips found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        frames_dir, records = self.samples[index]
        frames = []
        for record in records:
            image_path = frames_dir / record["frame"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        target = np.asarray([float(records[-1][key]) for key in VISUAL_TARGET_KEYS], dtype=np.float32)
        return torch.from_numpy(np.stack(frames)), torch.from_numpy(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the temporal screen-state reconstruction model.")
    parser.add_argument("--data-dir", default="data/live_vision")
    parser.add_argument("--output", default="checkpoints/visual_state_v1.pt")
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = RecordedClipDataset(args.data_dir, clip_length=args.clip_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TemporalStateNet().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for frames, targets in loader:
            frames = frames.to(args.device)
            targets = targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(frames)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * frames.shape[0]
            samples += frames.shape[0]
        print(f"epoch={epoch} loss={total_loss / max(1, samples):.6f}", flush=True)
    model.save(args.output)
    print(f"saved={args.output} clips={len(dataset)} device={args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

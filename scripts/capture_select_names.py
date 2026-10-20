"""Re-learns the character select name templates (data/character_select_names.npz) live.

    python scripts/capture_select_names.py [--delay 8]

At character select with the P2 cursor active (P1 already picked), your controller unplugged.
The P2 cursor walks the whole grid; every hovered name is read from the right-hand side (where
the existing templates read it, at distance ~0) and its letter image is captured from a wide
crop, so long names (ARMOR KING, YOSHIMITSU...) are stored whole. Names never seen keep their
old template.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_calibrate_distance import fresh_frame  # noqa: E402
from t8_agent.live.menu_automation import (NAME_ANCHORS, NAME_BAND, TEMPLATES, _letters,  # noqa: E402
                                           load_templates, pad_presser, read_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--delay", type=float, default=8)
    args = parser.parse_args()
    import dxcam
    from t8_agent.io.controller_backend import VGamepadInputBackend
    camera = dxcam.create(output_idx=0)
    pad = VGamepadInputBackend(facing=1, asynchronous=True, hold_timeout=0.25)
    press = pad_presser(pad)
    templates = load_templates()
    captured: dict[str, np.ndarray] = {}
    x0 = 1000
    try:
        time.sleep(args.delay)
        press("start", after=1.5)  # answers the "Press Menu" prompt of a new pad
        for _ in range(3):
            press("up")
        for row in range(3):
            for _ in range(17):
                time.sleep(0.2)
                for _ in range(4):  # a few frames: the fighter model behind may touch a letter
                    frame = fresh_frame(camera, 1.0)
                    if frame is None:  # nothing changed on screen
                        continue
                    name, distance = read_name(frame, 2, templates)
                    top, bottom = NAME_BAND
                    found = _letters(frame[top:bottom, x0:, ::-1], "right", NAME_ANCHORS[2][1] - x0)
                    if name and distance < 0.05 and found is not None and found[1] is None:
                        old = captured.get(name)
                        if old is None or found[0].shape[1] > old[0].shape[1]:
                            captured[name] = found[0], frame[top:bottom, x0:, ::-1].copy()
                        break
                    time.sleep(0.1)
                press("right")
            press("down")
    finally:
        pad.release_all()
        pad.close()
    print(f"captured {len(captured)} names: {sorted(captured)}")
    print(f"not seen (old template kept): {sorted(set(templates) - set(captured))}")
    out = {}
    for name, (image, cut) in templates.items():
        if name in captured:
            image, cut = captured[name][0], None
        flag = np.zeros((1, image.shape[1]), np.float32)
        flag[0, 0] = {"left": 1.0, "right": 2.0}.get(cut, 0.0)
        out[name.replace("-", "_")] = np.vstack([image, flag])
    np.savez_compressed(TEMPLATES, **out)
    print(f"saved {TEMPLATES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

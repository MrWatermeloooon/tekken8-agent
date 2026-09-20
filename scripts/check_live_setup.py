from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely verify CUDA inference, DXcam capture, and a neutral virtual Xbox controller."
    )
    parser.add_argument("--screen-config", type=Path, default=Path("config/live_screen.yaml"))
    parser.add_argument("--capture-attempts", type=int, default=30)
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--skip-controller", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA")
    print(
        f"cuda_ok torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)}"
    )

    from t8_agent.io.screen_backend import DxcamScreenStateBackend

    config_path = args.screen_config if args.screen_config.exists() else None
    backend = DxcamScreenStateBackend(config_path=config_path)
    state = None
    try:
        for _ in range(max(1, args.capture_attempts)):
            state = backend.read()
            if state.raw and state.raw.get("capture_valid"):
                break
            time.sleep(0.02)
    finally:
        backend.close()
    if state is None or not state.raw or not state.raw.get("capture_valid"):
        raise RuntimeError("DXcam opened but did not return a valid desktop frame")
    calibrated = bool(
        state.raw.get("has_health_calibration") and state.raw.get("has_position_calibration")
    )
    print(
        f"capture_ok size={state.raw['screen_width']}x{state.raw['screen_height']} "
        f"processing_ms={state.raw['capture_processing_ms']:.2f} calibrated={calibrated} "
        f"source={backend.config.get('calibration_source', 'unspecified')}"
    )
    if args.require_calibration and not calibrated:
        raise RuntimeError(f"screen regions are not calibrated in {args.screen_config}")

    if args.skip_controller:
        print("controller_skipped")
        return 0

    from t8_agent.io.controller_backend import VGamepadInputBackend
    try:
        controller = VGamepadInputBackend(tap_seconds=0.0, dash_gap_seconds=0.0)
    except Exception as exc:
        raise RuntimeError(
            "the virtual controller could not attach; install/start ViGEmBus and retry"
        ) from exc
    # This deliberately sends no gameplay input. It only asks the virtual
    # device to report a fully neutral state, proving the driver is available.
    controller.release_all()
    print("controller_ok device=virtual_xbox_360 state=neutral")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

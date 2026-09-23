"""One-off converter: repackages a V2-format (.t8ppo, header version 2)
checkpoint into the V3 contract format the live-play loader now requires.

This does not touch the weights or optimizer-state payload at all -- it is
byte-for-byte identical. It only inserts the new CheckpointContract block
(using ActorCriticConfig's own defaults, which is exactly what a rebuilt
trainer would have written for a run that never opted into the parametric
action head) between the existing header and integrity blocks, and bumps the
version field from 2 to 3.

This exists because the checkpoint file format moved out from under an
in-progress training run: the run's t8_v2_train.exe binary was built before
the V3 contract landed in the source, so its checkpoints are permanently
stuck at V2 -- they were never going to become V3 no matter how long that
run continued. Converting is the only way to load them with the current
live-play code without retraining.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

_HEADER = struct.Struct("<8sIIIIQQ")
_CONTRACT = struct.Struct("<64s24s32s32sII")
_INTEGRITY = struct.Struct("<QQ")
_MAGIC = b"T8V2PPO\0"

# Matches t8::v2::ActorCriticConfig's own defaults in include/t8_v2/ppo.hpp --
# i.e. exactly what a rebuilt trainer would stamp onto a non-parametric,
# fixed-24-action run like this one.
_DEFAULT_CATALOG_SHA256 = "0" * 64
_DEFAULT_ROSTER_VERSION = "compatibility"
_DEFAULT_OBSERVATION_CONTRACT = "v2-observation"
_DEFAULT_ACTION_CONTRACT = "fixed-24-compatibility"
_DEFAULT_ACTION_FEATURE_SIZE = 0
_DEFAULT_UNIVERSAL_ACTION_COUNT = 18


def _pad(text: str, size: int) -> bytes:
    encoded = text.encode("ascii")
    if len(encoded) > size:
        raise ValueError(f"contract field {text!r} does not fit in {size} bytes")
    return encoded + b"\0" * (size - len(encoded))


def read_contract(reference: Path) -> bytes:
    """Contract block of a V3/V4 checkpoint written by the current trainer."""
    raw = reference.read_bytes()
    magic, version, *_ = _HEADER.unpack_from(raw)
    if magic != _MAGIC or version < 3:
        raise ValueError(f"--contract-from needs a V3+ checkpoint: {reference}")
    return raw[_HEADER.size : _HEADER.size + _CONTRACT.size]


def convert(source: Path, destination: Path, contract: bytes | None = None) -> None:
    raw = source.read_bytes()
    if len(raw) < _HEADER.size + _INTEGRITY.size:
        raise ValueError(f"truncated V2 checkpoint: {source}")
    magic, version, observations, actions, hidden, optimizer_step, parameter_count = (
        _HEADER.unpack_from(raw)
    )
    if magic != _MAGIC:
        raise ValueError(f"not a T8V2PPO checkpoint: {source}")
    if version != 2:
        raise ValueError(f"expected a V2 checkpoint (got version {version}): {source}")

    integrity_bytes = raw[_HEADER.size : _HEADER.size + _INTEGRITY.size]
    payload = raw[_HEADER.size + _INTEGRITY.size :]
    payload_bytes, _checksum = _INTEGRITY.unpack(integrity_bytes)
    if len(payload) != payload_bytes:
        raise ValueError(f"V2 checkpoint payload size mismatch: {source}")

    new_header = _HEADER.pack(
        magic, 3, observations, actions, hidden, optimizer_step, parameter_count)
    new_contract = contract if contract is not None else _CONTRACT.pack(
        _pad(_DEFAULT_CATALOG_SHA256, 64),
        _pad(_DEFAULT_ROSTER_VERSION, 24),
        _pad(_DEFAULT_OBSERVATION_CONTRACT, 32),
        _pad(_DEFAULT_ACTION_CONTRACT, 32),
        _DEFAULT_ACTION_FEATURE_SIZE,
        _DEFAULT_UNIVERSAL_ACTION_COUNT,
    )

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    destination.write_bytes(new_header + new_contract + integrity_bytes + payload)
    print(
        f"wrote {destination} (observations={observations} actions={actions} "
        f"hidden={hidden} optimizer_step={optimizer_step})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--contract-from", type=Path, default=None,
        help="Copy the contract block from this V3/V4 checkpoint (for example one written by the "
             "current trainer with the same options) so the C++ loader accepts the result.")
    args = parser.parse_args()
    convert(args.source, args.destination,
            read_contract(args.contract_from) if args.contract_from else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

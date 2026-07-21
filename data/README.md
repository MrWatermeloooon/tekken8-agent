# V1 reference data used by V2

These files are copied from V1 commit
`7347212d036fcd6212fcf81864f6b2c96df0a524`, making the V2 branch independent
of the V1 branch at build and runtime:

- `universal_actions.yaml` is the authoritative movement, defense, attack, and
  system-action catalog.
- `characters/jun.yaml` is the full Jun frame-data and curriculum reference.

The CUDA simulator intentionally exposes a stable 24-action abstraction. Six actions map to
distilled single-move combat specifications; the remaining actions cover movement and defense.
The full 149-entry Jun table is reference/calibration data, not silently loaded into kernels at
runtime. Expanding the action boundary requires a new versioned contract, regenerated CUDA data,
CPU/GPU parity fixtures, controller mappings, and new policies—never an in-place index change.

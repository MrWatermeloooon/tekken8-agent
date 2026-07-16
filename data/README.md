# V1 reference data used by V2

These files are copied from V1 commit
`7347212d036fcd6212fcf81864f6b2c96df0a524`, making the V2 branch independent
of the V1 branch at build and runtime:

- `universal_actions.yaml` is the authoritative movement, defense, attack, and
  system-action catalog.
- `characters/jun.yaml` is the full Jun frame-data and curriculum reference.

The current CUDA simulator implements the smaller frozen training boundary in
`contracts/v1_contract.json`. The complete YAML catalogs are preserved for the
planned mechanics expansion and must not silently change frozen action indices.

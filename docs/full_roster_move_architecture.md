# Full-Roster Move Architecture

## Implemented foundation

`tools/compile_full_move_catalog.py` compiles the saved roster snapshots into a checksummed JSON
document and compact native CSV. It preserves 6,393 source rows for 41 characters. Bob has no
source rows and is blocked; Roger Jr. remains outside the released roster.

Each move has a stable `(character, source_index)` identity, parsed command steps, legal-state
requirements, move features, provenance, validation status, and explicit `null` values for combat
measurements absent from the source. Contradictory data is preserved and marked as blocking. It is
never silently reordered or replaced with a generic move.

The CUDA actor-critic supports a parametric head. Its state encoder emits a 32-value query and a
state-only value. Candidate logits are scaled dot products between that query and the selected
character's move features. PPO gradients reduce through those features into the shared encoder.

Checkpoint format v3 binds weights to the catalog SHA-256, roster version, observation contract,
action contract, feature size, and universal-action count. Trainer state v6 also binds exact resume
to the catalog and learner character. Older artifacts fail with an explicit incompatibility error.

## Character gate

```powershell
python tools\check_character_gate.py jun
```

A character remains disabled until all of the following are true:

1. Every source command parses without unresolved shorthand.
2. Source frame ranges are internally consistent.
3. Active frames, range, tracking, pushback, and collision are measured.
4. Every move passes offline Practice command validation.
5. Hit, block, counter-hit, whiff, transition, and resource scenarios pass.
6. Complete-catalog CPU tests and randomized CUDA parity pass.
7. Midscreen, wall, Heat, and counter-hit routes pass.

Persistent manual results belong in `data/validation/<character>.json` and are written atomically
by `scripts/validate_move_commands.py`.

## Current boundary

The old six-slot simulator is retained as a compatibility rollout path while exact full-move combat
measurements are collected. It must not be described as full-move training. The full catalog,
parametric CUDA PPO, controller executor, and strict gates are implemented; enabling Jun training
is blocked by the gate report instead of placeholder physics.

The scalar full-combat oracle now covers symmetric trades, posture, stances/resources, Heat/Rage,
recoverable health, armor, crush, parry/reversal, throws, axis/tracking, launches, scaling, tornado,
wall splats, and stage breaks using validated synthetic fixtures. The next engine milestone is
binding validated Jun catalog records to that oracle, then implementing the CUDA mirror and
randomized parity. Only then should the parametric head become the default rollout path and
`--learner-character all` begin sampling enabled characters.

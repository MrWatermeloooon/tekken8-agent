# Full-roster curriculum

## Coverage

The learner is always Jun. The opponent catalog contains all 42 playable fighters in the official
2026-07-21 roster snapshot. Every character receives exactly 50 profiles from ten archetypes and
five variations: rushdown, defensive turtle, whiff punisher, keep-out, throw-heavy, low-heavy,
string knowledge, counter-hit, movement specialist, and character specialist.

Profiles are distributions, not fixed scripts. They carry aggression, reaction range, input error,
approach/backdash/sidestep weights, lows, throws, delay, stance, Heat, punishment, throw break, and
low block parameters. Python episode variants and the CUDA opponent kernel both apply deterministic
seeded jitter.

## GPU path

The hot loop keeps profile tables and assignments in device memory. Each fighter lane also carries
a character ID. Six stable attack actions look up that character's distilled move in CUDA constant
memory, while movement and defense keep the versioned 24-action contract. Done-lane resets preserve
the IDs. A regression test assigns two different jab damages and proves both the combat lookup and
reset persistence execute on the GPU.

Visual roster policies consume 95 values:

- 13 screen-compatible base features;
- 8 character-embedding values;
- 10 archetype one-hot values;
- 8 decisions x 8 history values: move/action ID, animation phase, stance, hit level, repeated-action
  delay, outcome, distance, and side movement.

The privileged equivalent is 101 values (19 + the same 82-value matchup context). Temporal state is
saved and restored with simulator, scheduler, profile assignment, opponent action, and RNG state.

## Curriculum and evaluation

1. Jun fundamentals: easier variations and core archetypes from fundamentals-tagged characters.
2. Character groups: rotating rushdown, stance, grappler, keep-out, evasive, and specialist groups.
3. Full roster: all 2,100 profiles.
4. Adversarial league: the full roster weighted toward weak, uncertain, regressing, and exploitative
   cells, plus typed historical-checkpoint, exploit-policy, and human-failure league entries.

The scheduler avoids training only on the hardest profile when a matchup is extremely weak by
up-weighting easier variations. Evaluation is indexed by character and archetype and supports
outcomes, punishment, throw break, low defense, string interruption, sidestep, Heat defense, wall
escape, matchup Elo, and catastrophic forgetting.

## Reproduction

```powershell
python tools\import_roster_frame_data.py --allow-missing
python tools\generate_opponent_catalog.py --data-root data
python -m pytest tests\test_roster_system.py tests\test_v2_live_policy.py -q `
  --basetemp build\pytest-tmp -p no:cacheprovider
ctest --test-dir build -C Release --output-on-failure
```

The importer never invents missing source data. Bob remains explicitly marked unavailable in the
source snapshot, and the generated manifest hashes both runtime CSVs.

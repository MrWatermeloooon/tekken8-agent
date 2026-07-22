# V2 roster and matchup data

`roster.yaml` snapshots the 42 playable fighters shown on the official Tekken 8 fighters page on
2026-07-21. Roger Jr. is recorded separately as announced/unreleased and is not silently treated as
playable. `opponent_archetypes.yaml` defines the deterministic 10 x 5 profile generator.

`characters/*.yaml` was imported from the public TekkenDocs API with retrieval dates, source URLs,
and source hashes. The TekkenDocs project explicitly permits exposed data reuse with attribution.
The API supplied 41 roster entries; Bob is an explicit unavailable marker, and only his six-slot
simulation table uses documented abstract fallback values.

Generated files:

- `generated/opponent_profiles.csv`: 2,100 profiles, exactly 50 per character.
- `generated/character_move_specs.csv`: 42 x 6 combat slots loaded by CUDA.
- `generated/manifest.json`: counts and SHA-256 hashes.
- `character_modules/*/matchup.yaml`: punishable moves, strings/gaps, duckable highs, sidestep,
  stances, throws, lows, power crushes, Heat threats, range, launch punishment, and Jun responses.

Regenerate deterministically with:

```powershell
python tools\import_roster_frame_data.py --allow-missing
python tools\generate_opponent_catalog.py --data-root data
```

Sources: [official roster](https://tekken.com/fighters),
[TekkenDocs API](https://tekkendocs.com/api/t8/characters), and
[TekkenDocs repository/attribution note](https://github.com/pbruvoll/tekkendocs).

The simulator retains a stable 24-action boundary. Six attack actions select character-specific
distilled moves; the remaining actions cover movement and defense. Expanding that boundary requires
a new versioned policy contract, controller mapping, CUDA fixtures, and parity tests.

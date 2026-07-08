# Character Curriculum

The bot should learn one character deeply before we add the whole roster. The
right order is:

1. Build a real move and combo catalog for Jun.
2. Train a Jun policy on core pokes, punishers, lows, launchers, and routes.
3. Add matchup opponents one character at a time.
4. Expand self-play pools by character and skill level.
5. Only then train against broad roster sampling.

## Why Not Add Everyone At Once

Every character adds more moves, stance states, punish rules, combo routes, and
matchup-specific defense. If we add the full roster before Jun is stable, the
agent will mostly learn noise. A better curriculum is:

- Jun mirror.
- Jun versus one simple opponent style.
- Jun versus one real character catalog.
- Jun versus a small rotating pool.
- Full roster.

## Catalog Rules

Character data lives under `data/characters/`. Each catalog has:

- `moves`: factual frame-data entries and training tags.
- `combos`: routes by starter and purpose.
- `tier`: `core` for moves the bot should learn first, `specialist` for later.
- `source_url`: where the data came from.

Jun's first catalog is seeded from TekkenDocs:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\character_catalog.py data\characters\jun.yaml
```

The current Jun catalog is a curated core seed, not the final exhaustive list.
Next we should import/verify every TekkenDocs Jun entry, then record practical
combo routes in Practice mode.

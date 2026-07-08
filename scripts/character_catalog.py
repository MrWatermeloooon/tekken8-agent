from __future__ import annotations

import argparse
from pathlib import Path

from t8_agent.data.character_catalog import load_character_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Tekken character catalog.")
    parser.add_argument("catalog", nargs="?", default="data/characters/jun.yaml")
    args = parser.parse_args()

    catalog = load_character_catalog(Path(args.catalog))
    print(f"character={catalog.display_name} game={catalog.game}")
    print(f"source={catalog.source_url}")
    print(f"moves={len(catalog.moves)} core_moves={len(catalog.core_moves)} combos={len(catalog.combos)}")
    print("core:")
    for move in catalog.core_moves:
        tags = ",".join(move.tags)
        print(f"  {move.move_id:<18} {move.command:<10} {move.startup:<8} block={move.block:<8} hit={move.hit:<8} {tags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

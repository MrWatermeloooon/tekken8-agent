from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from t8_agent.moves.catalog import load_compiled_catalog  # noqa: E402
from t8_agent.moves.validation import evaluate_character_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Report strict full-move training gates.")
    parser.add_argument("character", help="Released roster slug or 'all'.")
    parser.add_argument("--catalog", type=Path,
                        default=REPO_ROOT / "data/generated/full_move_catalog.json")
    parser.add_argument("--validation-root", type=Path,
                        default=REPO_ROOT / "data/validation")
    args = parser.parse_args()
    catalog = load_compiled_catalog(args.catalog)
    slugs = [row["slug"] for row in catalog.characters]
    selected = slugs if args.character == "all" else [args.character]
    reports = [evaluate_character_gate(catalog, slug, args.validation_root) for slug in selected]
    print(json.dumps({"ready": all(report.ready for report in reports),
                      "characters": [report.to_dict() for report in reports]}, indent=2))
    return 0 if all(report.ready for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())

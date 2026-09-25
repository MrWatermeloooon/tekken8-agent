from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from t8_agent.moves.catalog import compile_catalog, write_catalog  # noqa: E402


def native_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile all documented Tekken 8 moves into the V2 contract.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/generated/full_move_catalog.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "data/generated/full_move_validation.csv")
    parser.add_argument("--native-csv", type=Path, default=REPO_ROOT / "data/generated/full_move_catalog.csv")
    args = parser.parse_args()
    catalog = compile_catalog(args.data_root)
    write_catalog(catalog, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "character_id", "character", "data_status", "moves", "parsed", "needs_review",
            "practice_validation", "training_enabled", "source_url",
        ], lineterminator="\n")
        writer.writeheader()
        for character in catalog.characters:
            writer.writerow({
                "character_id": character["id"], "character": character["slug"],
                "data_status": character["data_status"], "moves": character["move_count"],
                "parsed": character["parsed_move_count"], "needs_review": character["review_move_count"],
                "practice_validation": character["practice_validation"],
                "training_enabled": str(character["training_enabled"]).lower(),
                "source_url": character["source_url"],
            })
    with args.native_csv.open("w", newline="", encoding="utf-8") as handle:
        feature_columns = [f"feature_{index}" for index in range(32)]
        fields = [
            "schema_version", "catalog_sha256", "roster_version",
            "character_id", "local_id", "source_index", "stable_id", "name", "command",
            "parser_status", "hit_level", "damage", "startup_min", "startup_max",
            "recovery_min", "recovery_max", "block_min", "block_max", "source_consistency",
            "validation_issues", "mechanic_flags",
        ] + feature_columns
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for move in catalog.moves:
            mechanics = move["mechanics"]
            flags = [key for key, value in mechanics.items() if isinstance(value, bool) and value]
            row = {
                "schema_version": catalog.document["schema_version"],
                "catalog_sha256": catalog.catalog_sha256,
                "roster_version": catalog.document["roster_as_of"],
                "character_id": move["character_id"], "local_id": move["local_id"],
                "source_index": move["source_index"], "stable_id": move["stable_id"],
                "name": native_text(move["name"]), "command": native_text(move["command"]["raw"]),
                "parser_status": move["command"]["parser_status"],
                "hit_level": move["hit_levels"][0] if move["hit_levels"] else "unknown",
                "damage": sum(move["damage"]),
                "startup_min": "" if move["startup"] is None else move["startup"]["min"],
                "startup_max": "" if move["startup"] is None else move["startup"]["max"],
                "recovery_min": "" if move["recovery"] is None else move["recovery"]["min"],
                "recovery_max": "" if move["recovery"] is None else move["recovery"]["max"],
                "block_min": "" if move["block_advantage"] is None else move["block_advantage"]["min"],
                "block_max": "" if move["block_advantage"] is None else move["block_advantage"]["max"],
                "source_consistency": move["validation"]["source_consistency"],
                "validation_issues": "|".join(move["validation"]["issues"]),
                "mechanic_flags": "|".join(flags),
            }
            row.update({column: move["action_features"][index]
                        for index, column in enumerate(feature_columns)})
            writer.writerow(row)
    print(json.dumps({
        "characters": len(catalog.characters), "moves": len(catalog.moves),
        "catalog_sha256": catalog.catalog_sha256, "output": str(args.output),
        "report": str(args.report), "native_csv": str(args.native_csv),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

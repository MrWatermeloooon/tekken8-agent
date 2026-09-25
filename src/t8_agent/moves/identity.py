"""Stable move identity across source patches.

A move's stable ID is ``<slug>:<number>``. The number is pinned to the move's
source key in ``data/identity/<slug>.json``, so a source update that inserts,
removes, or reorders moves never renumbers the others (validation ledgers,
measurements, and checkpoints are keyed by stable ID). New moves get the next
unused number; removed moves keep theirs as retired and it is never reused.

The source key is the source's own ID (TekkenDocs/Wavu ``source_id``). A few
sources repeat an ID on distinct rows (alternate versions of one move); the
second and later repeats are keyed ``<source_id>#<n>`` in source order.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

IDENTITY_SCHEMA_VERSION = 1


def source_keys(rows: list[dict[str, Any]]) -> list[str]:
    seen: Counter[str] = Counter()
    keys = []
    for row in rows:
        source_id = str(row.get("source_id") or row.get("id") or "")
        if not source_id:
            raise ValueError(f"move without a source_id: {row.get('command')!r}")
        seen[source_id] += 1
        keys.append(source_id if seen[source_id] == 1 else f"{source_id}#{seen[source_id]}")
    return keys


def identity_path(data_root: str | Path, slug: str) -> Path:
    return Path(data_root) / "identity" / f"{slug}.json"


def load_identity(data_root: str | Path, slug: str) -> dict[str, Any] | None:
    path = identity_path(data_root, slug)
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != IDENTITY_SCHEMA_VERSION or document.get("character") != slug:
        raise ValueError(f"invalid move identity registry: {path}")
    return document


def write_identity(data_root: str | Path, registry: dict[str, Any]) -> None:
    path = identity_path(data_root, str(registry["character"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bootstrap_identity(slug: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A registry that reproduces the current IDs (the source's move numbers)."""
    moves = {key: int(row["source_index"]) for key, row in zip(source_keys(rows), rows)}
    if len(set(moves.values())) != len(moves):
        raise ValueError(f"{slug}: source move numbers are not unique")
    return {"schema_version": IDENTITY_SCHEMA_VERSION, "character": slug,
            "next_number": max(moves.values(), default=0) + 1, "moves": moves, "retired": {}}


def stable_numbers(slug: str, rows: list[dict[str, Any]], registry: dict[str, Any]) -> list[int]:
    """The pinned number of every row; fails on rows the registry does not know."""
    keys = source_keys(rows)
    unknown = [key for key in keys if key not in registry["moves"]]
    if unknown:
        raise ValueError(f"{slug}: moves without a stable ID (run tools/patch_update.py --apply): {unknown[:5]}")
    return [int(registry["moves"][key]) for key in keys]


def update_identity(registry: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Assigns numbers to new keys and retires missing ones. Returns (registry, added, retired)."""
    keys = source_keys(rows)
    updated = json.loads(json.dumps(registry))
    added = [key for key in keys if key not in updated["moves"]]
    retired = [key for key in updated["moves"] if key not in set(keys)]
    for key in retired:
        updated["retired"][key] = updated["moves"].pop(key)
    for key in added:
        if key in updated["retired"]:
            # A move that comes back keeps its old number.
            updated["moves"][key] = updated["retired"].pop(key)
        else:
            updated["moves"][key] = int(updated["next_number"])
            updated["next_number"] = int(updated["next_number"]) + 1
    return updated, added, retired

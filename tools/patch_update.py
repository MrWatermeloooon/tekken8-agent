"""Game-patch update workflow for a character's move source.

Diffs a new source snapshot against the saved one and, with --apply, updates
the data and invalidates everything that depended on a changed move:

  * the saved snapshot is archived under data/characters/history/<slug>/;
  * the move identity registry (data/identity/<slug>.json) gives new moves
    new stable IDs and retires removed ones, so no other move is renumbered;
  * Practice validations (data/validation/<slug>.json) of changed moves become
    "stale" and of removed moves "retired", keeping the previous record;
    scenario gates become stale on any change, and route gates when a route
    uses a changed or removed move;
  * measurements (data/measurements/<slug>.yaml) of changed moves are marked
    stale, so the binder treats them as missing;
  * corrections whose corrected field changed upstream become "needs_review",
    or "resolved_upstream" when the new source value equals the correction;
  * a patch record is written to data/patches/<slug>/.

    python tools\\patch_update.py --bootstrap                 # create identity registries
    python tools\\patch_update.py jun --fetch                 # dry run against TekkenDocs
    python tools\\patch_update.py jun --snapshot new.yaml     # dry run against a file
    python tools\\patch_update.py jun --fetch --apply         # update and invalidate

Afterwards rebuild the catalog and bindings (tools/compile_full_move_catalog.py,
tools/export_full_combat_bindings.py).
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from t8_agent.moves.identity import (  # noqa: E402
    bootstrap_identity, load_identity, source_keys, update_identity, write_identity,
)

COMPARED_FIELDS = ("command", "name", "hit_level", "damage", "startup", "recovery", "block", "hit",
                   "counter_hit", "notes")


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Move-level diff keyed by source key: added, removed, changed fields, renumbered."""
    old_rows, new_rows = list(old.get("moves") or []), list(new.get("moves") or [])
    old_by_key = dict(zip(source_keys(old_rows), old_rows))
    new_by_key = dict(zip(source_keys(new_rows), new_rows))
    changed: dict[str, dict[str, Any]] = {}
    for key in old_by_key.keys() & new_by_key.keys():
        before, after = old_by_key[key], new_by_key[key]
        fields = {}
        for field in COMPARED_FIELDS:
            a, b = str(before.get(field) or ""), str(after.get(field) or "")
            if a == b:
                continue
            if field == "notes":
                a_lines = [line.strip("* ").strip() for line in a.splitlines() if line.strip("* ").strip()]
                b_lines = [line.strip("* ").strip() for line in b.splitlines() if line.strip("* ").strip()]
                if a_lines == b_lines:
                    continue  # whitespace only
                fields[field] = {"removed": [line for line in a_lines if line not in b_lines],
                                 "added": [line for line in b_lines if line not in a_lines]}
            else:
                fields[field] = {"old": a, "new": b}
        if fields:
            changed[key] = fields
    renumbered = sorted(key for key in old_by_key.keys() & new_by_key.keys()
                        if old_by_key[key].get("source_index") != new_by_key[key].get("source_index"))
    return {
        "added": sorted(new_by_key.keys() - old_by_key.keys()),
        "removed": sorted(old_by_key.keys() - new_by_key.keys()),
        "changed": dict(sorted(changed.items())),
        "source_renumbered": renumbered,
        "old_retrieved_at": old.get("retrieved_at"), "new_retrieved_at": new.get("retrieved_at"),
        "old_sha256": old.get("source_sha256"), "new_sha256": new.get("source_sha256"),
    }


def fetch_snapshot(slug: str, timeout: float) -> dict[str, Any]:
    import import_roster_frame_data as importer
    from t8_agent.roster.catalog import load_catalog

    character = next(row for row in load_catalog(REPO_ROOT / "data").characters if row.slug == slug)
    url = f"{importer.API_ROOT}/{slug}/framedata"
    return importer._convert_payload(slug, character.name, url, importer._download_json(url, timeout))


def summarize(slug: str, diff: dict[str, Any]) -> str:
    lines = [f"{slug}: {len(diff['added'])} added, {len(diff['removed'])} removed, "
             f"{len(diff['changed'])} changed ({diff['old_retrieved_at']} -> {diff['new_retrieved_at']})"]
    for key in diff["added"]:
        lines.append(f"  + {key}")
    for key in diff["removed"]:
        lines.append(f"  - {key}")
    for key, fields in diff["changed"].items():
        lines.append(f"  ~ {key}")
        for field, change in fields.items():
            if field == "notes":
                lines += [f"      notes -{line}" for line in change["removed"]]
                lines += [f"      notes +{line}" for line in change["added"]]
            else:
                lines.append(f"      {field}: {change['old']!r} -> {change['new']!r}")
    if diff["source_renumbered"]:
        lines.append(f"  (source move numbers shifted for {len(diff['source_renumbered'])} moves; stable IDs are unaffected)")
    return "\n".join(lines)


def _stale(record: Any, reason: str, status: str = "stale") -> dict[str, Any]:
    previous = record if isinstance(record, dict) else {"status": record}
    history = list(previous.get("history") or [])
    history.append({key: value for key, value in previous.items() if key != "history"})
    return {"status": status, "reason": reason, "history": history}


def apply_patch(slug: str, new: dict[str, Any], data_root: Path, today: str | None = None) -> dict[str, Any]:
    today = today or date.today().isoformat()
    source_path = data_root / "characters" / f"{slug}.yaml"
    old = read_yaml(source_path)
    diff = diff_snapshots(old, new)
    registry = load_identity(data_root, slug) or bootstrap_identity(slug, list(old.get("moves") or []))
    old_numbers = dict(registry["moves"])
    updated, added, retired = update_identity(registry, list(new.get("moves") or []))
    # A renamed move is the same move; any other change invalidates what was checked against it.
    substantive = [key for key, fields in diff["changed"].items() if set(fields) - {"name"}]
    affected = {key: f"{slug}:{old_numbers[key]}" for key in substantive + diff["removed"]}
    reason = f"source changed on {today} (retrieved {new.get('retrieved_at')})"
    invalidated: dict[str, Any] = {"validations": [], "measurements": [], "scenarios": [], "routes": [],
                                   "corrections": {}}

    # Archive, then write the new snapshot and identities.
    history = data_root / "characters" / "history" / slug
    history.mkdir(parents=True, exist_ok=True)
    archived = history / f"{old.get('retrieved_at', 'unknown')}_{str(old.get('source_sha256', ''))[:12]}.yaml"
    archived.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_path.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=False), encoding="utf-8")
    write_identity(data_root, updated)

    changed_or_removed = set(substantive) | set(diff["removed"])
    any_change = bool(diff["added"] or changed_or_removed)

    ledger_path = data_root / "validation" / f"{slug}.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        moves = ledger.setdefault("moves", {})
        for key, stable_id in affected.items():
            if stable_id in moves:
                moves[stable_id] = _stale(moves[stable_id], reason, "retired" if key in diff["removed"] else "stale")
                invalidated["validations"].append(stable_id)
        if any_change:
            for name, value in list((ledger.get("scenarios") or {}).items()):
                if (value.get("status") if isinstance(value, dict) else value) == "pass":
                    ledger["scenarios"][name] = _stale(value, reason)
                    invalidated["scenarios"].append(name)
        route_file = data_root / "character_modules" / slug / "routes.yaml"
        if route_file.exists():
            old_rows = list(old.get("moves") or [])
            key_by_command = {str(row.get("command")): key for key, row in zip(source_keys(old_rows), old_rows)}
            for route in (read_yaml(route_file).get("routes") or []):
                commands = [str(step["input"] if isinstance(step, dict) else step) for step in route.get("steps") or []]
                if any(key_by_command.get(command) in changed_or_removed for command in commands):
                    category = str(route["category"])
                    routes = ledger.setdefault("routes", {})
                    if category in routes and (routes[category].get("status") if isinstance(routes[category], dict)
                                               else routes[category]) != "stale":
                        routes[category] = _stale(routes[category], f"{reason}; route {route['name']} uses a changed move")
                        invalidated["routes"].append(category)
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    measurement_path = data_root / "measurements" / f"{slug}.yaml"
    if measurement_path.exists():
        document = read_yaml(measurement_path)
        for stable_id in affected.values():
            entry = (document.get("moves") or {}).get(stable_id)
            if entry is not None and not entry.get("stale"):
                entry["stale"] = reason
                invalidated["measurements"].append(stable_id)
        measurement_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    corrections_path = data_root / "corrections" / f"{slug}.yaml"
    if corrections_path.exists():
        text = corrections_path.read_text(encoding="utf-8")
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#")) if text.startswith("#") else ""
        document = yaml.safe_load(text) or {}
        new_rows = {str(row.get("source_id")): row for row in new.get("moves") or []}
        for entry in document.get("corrections") or []:
            source_id = str(entry.get("source_id"))
            if source_id not in new_rows:
                entry["status"] = "needs_review"
                entry["upstream_change"] = f"{reason}: the move was removed"
                invalidated["corrections"][source_id] = entry["status"]
                continue
            recorded = entry.get("source_fields") or {}
            for field, corrected in (entry.get("fields") or {}).items():
                upstream = str(new_rows[source_id].get(field) or "")
                if upstream == str(corrected):
                    entry["status"] = "resolved_upstream"
                    entry["upstream_change"] = f"{reason}: the source now reads {upstream!r}"
                elif field in recorded and upstream != str(recorded[field]):
                    entry["status"] = "needs_review"
                    entry["upstream_change"] = f"{reason}: {field} {recorded[field]!r} -> {upstream!r}"
                else:
                    continue
                invalidated["corrections"][source_id] = entry["status"]
        if invalidated["corrections"]:  # rewrite (losing inline comments) only when something changed
            corrections_path.write_text(header + yaml.safe_dump(document, sort_keys=False, allow_unicode=False),
                                        encoding="utf-8")

    record = {"character": slug, "applied_on": today, "archived_snapshot": archived.relative_to(data_root).as_posix(),
              "diff": diff, "identity": {"added": {key: updated["moves"][key] for key in added},
                                         "retired": {key: updated["retired"][key] for key in retired}},
              "invalidated": invalidated}
    patches = data_root / "patches" / slug
    patches.mkdir(parents=True, exist_ok=True)
    (patches / f"{today}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def bootstrap(data_root: Path) -> list[str]:
    created = []
    for path in sorted((data_root / "characters").glob("*.yaml")):
        slug = path.stem
        if load_identity(data_root, slug) is None:
            write_identity(data_root, bootstrap_identity(slug, list(read_yaml(path).get("moves") or [])))
            created.append(slug)
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("character", nargs="?")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--snapshot", type=Path, help="A new snapshot file in the saved format.")
    parser.add_argument("--fetch", action="store_true", help="Download the current TekkenDocs data.")
    parser.add_argument("--apply", action="store_true", help="Write the update and invalidate dependents.")
    parser.add_argument("--bootstrap", action="store_true", help="Create missing identity registries.")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.bootstrap:
        print(json.dumps({"identity_registries_created": bootstrap(args.data_root)}))
        return 0
    if not args.character or bool(args.snapshot) == args.fetch:
        parser.error("give a character and exactly one of --snapshot or --fetch")
    new = read_yaml(args.snapshot) if args.snapshot else fetch_snapshot(args.character, args.timeout)
    old = read_yaml(args.data_root / "characters" / f"{args.character}.yaml")
    diff = diff_snapshots(old, new)
    print(summarize(args.character, diff))
    if args.apply:
        record = apply_patch(args.character, new, args.data_root)
        print(json.dumps({"patch_record": f"data/patches/{args.character}/{record['applied_on']}.json",
                          "invalidated": record["invalidated"]}, indent=2))
    elif diff["added"] or diff["removed"] or diff["changed"]:
        print("dry run: nothing written (use --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

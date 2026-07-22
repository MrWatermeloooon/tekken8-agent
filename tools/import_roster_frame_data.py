from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import urllib.request
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from t8_agent.roster.catalog import load_catalog  # noqa: E402


CORE_COMMANDS = {
    "1", "1,2", "2", "4", "1+2", "df+1", "df+2", "df+3", "df+4",
    "f+2", "d+4", "db+3", "db+4", "b+2", "b+4", "uf+4", "f,F+2",
}
API_ROOT = "https://tekkendocs.com/api/t8"


def main() -> int:
    parser = argparse.ArgumentParser(description="Import full Tekken 8 roster frame data from TekkenDocs.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--only", nargs="*", default=None, help="Optional character slugs.")
    parser.add_argument("--allow-missing", action="store_true", help="Write an unavailable marker for missing API characters.")
    args = parser.parse_args()

    catalog = load_catalog(args.data_root)
    api_roster = _download_json(f"{API_ROOT}/characters", args.timeout)
    available = {str(row["id"]) for row in api_roster["characters"]}
    requested = set(args.only or [character.slug for character in catalog.characters])
    failures: list[str] = []
    imported = 0
    for character in catalog.characters:
        if character.slug not in requested:
            continue
        output = args.data_root / "characters" / f"{character.slug}.yaml"
        if character.slug not in available:
            failures.append(character.slug)
            if args.allow_missing:
                marker = {
                    "character_id": character.slug,
                    "display_name": character.name,
                    "game": "tekken_8",
                    "source_url": f"{API_ROOT}/{character.slug}/framedata",
                    "source_note": "No TekkenDocs frame-data endpoint existed when this roster snapshot was generated.",
                    "retrieved_at": date.today().isoformat(),
                    "data_status": "unavailable",
                    "stances": [],
                    "moves": [],
                }
                output.write_text(yaml.safe_dump(marker, sort_keys=False, allow_unicode=False), encoding="utf-8")
            continue
        url = f"{API_ROOT}/{character.slug}/framedata"
        payload = _download_json(url, args.timeout)
        document = _convert_payload(character.slug, character.name, url, payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=False), encoding="utf-8")
        imported += 1
        print(f"imported={character.slug} moves={len(document['moves'])}")
    if failures and not args.allow_missing:
        raise RuntimeError(f"frame data unavailable for: {', '.join(failures)}")
    print(f"characters_imported={imported} unavailable={','.join(failures) or 'none'}")
    return 0


def _download_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "tekken8-agent-v2-catalog/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def _convert_payload(slug: str, display_name: str, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    moves = [_convert_move(row) for row in payload.get("framesNormal", [])]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "character_id": slug,
        "display_name": display_name,
        "game": "tekken_8",
        "source_url": url,
        "source_note": "Full raw frame-data table imported from TekkenDocs; data reuse credited per its public repository license note.",
        "retrieved_at": date.today().isoformat(),
        "source_sha256": sha256(canonical).hexdigest(),
        "data_status": "available",
        "stances": [str(value) for value in payload.get("stances", [])],
        "moves": moves,
    }


def _convert_move(row: dict[str, Any]) -> dict[str, Any]:
    command = str(row.get("command") or "")
    hit_level = str(row.get("hitLevel") or "")
    notes = str(row.get("notes") or "")
    tags = [str(value) for value in (row.get("tags") or [])]
    for derived in _derived_tags(command, hit_level, notes):
        if derived not in tags:
            tags.append(derived)
    return {
        "id": _command_to_id(command),
        "source_id": str(row.get("wavuId") or ""),
        "source_index": int(row.get("moveNumber") or 0),
        "command": command,
        "name": str(row.get("name") or ""),
        "hit_level": hit_level,
        "damage": str(row.get("damage") or ""),
        "startup": str(row.get("startup") or ""),
        "recovery": str(row.get("recovery") or ""),
        "block": str(row.get("block") or ""),
        "hit": str(row.get("hit") or ""),
        "counter_hit": str(row.get("counterHit") or ""),
        "tier": "core" if command in CORE_COMMANDS else "specialist",
        "tags": tags,
        "notes": notes,
    }


def _command_to_id(command: str) -> str:
    text = command.replace("f,F", "ff").lower().replace("+", "p")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "move"


def _derived_tags(command: str, hit_level: str, notes: str) -> list[str]:
    blob = f"{command} {hit_level} {notes}".lower()
    tags: list[str] = []
    for needle, tag in (
        ("throw", "throw"), ("heat", "heat"), ("rage", "rage"),
        ("homing", "homing"), ("power crush", "power_crush"),
        ("low crush", "low_crush"), ("high crush", "high_crush"),
        ("tornado", "tornado"), ("balcony break", "balcony_break"),
        ("floor break", "floor_break"), ("parry", "parry"),
        ("sabaki", "parry"), ("launcher", "launcher"),
        ("transition", "stance_transition"), ("chip", "chip"),
    ):
        if needle in blob:
            tags.append(tag)
    lowered = hit_level.lower()
    if "l" in lowered: tags.append("low")
    if "m" in lowered: tags.append("mid")
    if "h" in lowered: tags.append("high")
    if "t" in lowered: tags.append("throw")
    return list(dict.fromkeys(tags))


if __name__ == "__main__":
    raise SystemExit(main())

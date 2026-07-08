from __future__ import annotations

import argparse
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml


CORE_COMMANDS = {
    "1",
    "1,2",
    "2",
    "4",
    "1+2",
    "df+1",
    "df+2",
    "df+3",
    "df+4",
    "f+2",
    "f+2,1+2",
    "d+4",
    "db+3",
    "db+4",
    "b+2",
    "b+2,1",
    "b+4",
    "uf+4",
    "f,F+2",
}

DEFAULT_COMBOS = [
    {
        "id": "df2_basic_wall_carry",
        "starter": "df2",
        "route": ["f2", "df1", "b2_1", "ff2"],
        "difficulty": "easy",
        "purpose": "launcher conversion",
        "notes": "Placeholder route for simulator curriculum; verify exact Tekken 8 route in practice.",
    },
    {
        "id": "uf4_basic_conversion",
        "starter": "uf4",
        "route": ["df1", "f2", "b2_1", "ff2"],
        "difficulty": "easy",
        "purpose": "low-crush launcher conversion",
        "notes": "Training route until combo calibration is recorded from the real game.",
    },
    {
        "id": "f2_whiff_punish_route",
        "starter": "f2",
        "route": ["f2_1p2", "ff2"],
        "difficulty": "medium",
        "purpose": "whiff punish and heat followup",
        "notes": "Use for spacing/whiff-punish curriculum.",
    },
]


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.cell_text: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.cell_text = []

    def handle_endtag(self, tag: str) -> None:
        if self.in_row and self.in_cell and tag in {"td", "th"}:
            text = " ".join("".join(self.cell_text).split())
            self.row.append(text)
            self.in_cell = False
        elif self.in_row and tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.in_row = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_text.append(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import TekkenDocs character frame data into a catalog YAML.")
    parser.add_argument("--url", default="https://tekkendocs.com/t8/jun")
    parser.add_argument("--html", default=None, help="Use a previously downloaded TekkenDocs HTML file.")
    parser.add_argument("--out", default="data/characters/jun.yaml")
    parser.add_argument("--character-id", default="jun")
    parser.add_argument("--display-name", default="Jun Kazama")
    args = parser.parse_args()

    html = Path(args.html).read_text(encoding="utf-8") if args.html else _download(args.url)
    rows = _parse_rows(html)
    moves = [_row_to_move(idx, row) for idx, row in enumerate(rows, start=1)]
    catalog: dict[str, Any] = {
        "character_id": args.character_id,
        "display_name": args.display_name,
        "game": "tekken_8",
        "source_url": args.url,
        "source_note": "Full raw frame-data table imported from TekkenDocs. Core tiers are our training curriculum labels.",
        "moves": moves,
        "combos": DEFAULT_COMBOS,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(f"wrote={out} moves={len(moves)}")
    return 0


def _download(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _parse_rows(html: str) -> list[list[str]]:
    parser = TableParser()
    parser.feed(html)
    rows = [row for row in parser.rows if len(row) == 8]
    if not rows or rows[0][:3] != ["Command", "Hit level", "Damage"]:
        raise RuntimeError("could not find TekkenDocs frame-data table")
    return rows[1:]


def _row_to_move(idx: int, row: list[str]) -> dict[str, Any]:
    command, hit_level, damage, startup, block, hit, counter_hit, notes = row
    move_id = _command_to_id(command)
    return {
        "id": move_id,
        "command": command,
        "hit_level": hit_level,
        "damage": damage,
        "startup": startup,
        "block": block,
        "hit": hit,
        "counter_hit": counter_hit,
        "tier": "core" if command in CORE_COMMANDS else "specialist",
        "tags": _tags_for_move(command, hit_level, notes),
        "notes": _clean_notes(notes),
        "source_index": idx,
    }


def _command_to_id(command: str) -> str:
    text = command.replace("f,F", "ff")
    text = text.replace("u_d", "sidestep")
    text = text.replace("Left throw", "left_throw")
    text = text.replace("Right throw", "right_throw")
    text = text.lower()
    for direction in ("df", "db", "uf", "ub", "ff", "f", "b", "d", "u"):
        text = text.replace(f"{direction}+", direction)
    text = text.replace("+", "p")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "move"


def _tags_for_move(command: str, hit_level: str, notes: str) -> list[str]:
    blob = f"{command} {hit_level} {notes}".lower()
    tags: list[str] = []
    for needle, tag in [
        ("throw", "throw"),
        ("heat", "heat"),
        ("rage", "rage"),
        ("homing", "homing"),
        ("power crush", "power_crush"),
        ("low crush", "low_crush"),
        ("high crush", "high_crush"),
        ("tornado", "tornado"),
        ("balcony break", "balcony_break"),
        ("floor break", "floor_break"),
        ("parry", "parry"),
        ("sabaki", "parry"),
        ("launcher", "launcher"),
        ("transition", "stance_transition"),
        ("chip", "chip"),
    ]:
        if needle in blob and tag not in tags:
            tags.append(tag)
    if "l" in hit_level.lower():
        tags.append("low")
    if "m" in hit_level.lower():
        tags.append("mid")
    if "h" in hit_level.lower():
        tags.append("high")
    return tags


def _clean_notes(notes: str) -> str:
    return notes.replace("*", "; ").strip("; ").strip()


if __name__ == "__main__":
    raise SystemExit(main())

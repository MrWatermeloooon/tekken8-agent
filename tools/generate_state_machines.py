"""Skeleton stance rules for characters that have no reviewed state machine yet.

    python tools/generate_state_machines.py            # write missing files
    python tools/generate_state_machines.py --check    # fail if any would change

For every roster character without data/character_modules/<slug>/state_machine.yaml,
writes one listing the character's stances: the stance abbreviations of its
TekkenDocs source (minus states the engine handles itself, such as H, ws, FC,
SS, BT, WR, hFC, lying-down positions, and malformed tokens) plus any stance a
move ends in ("r20 NSS") or is performed from ("NSS.1"). Each stance gets the
same stand-ins Jun's undocumented stances use: cannot guard, no time limit,
no automatic parry, all marked provisional. A file is generated only once;
after a person reviews it (status values other than provisional, parry
outcomes, pulses, a resource gauge), it is theirs and never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_STATES = {"H", "R", "WS", "SS", "FC", "HFC", "WR", "BT"}
# Lying-down positions (face up/down, feet toward/away) are postures, not stances.
GROUND_POSITIONS = {"FUFT", "FUFA", "FDFT", "FDFA", "FUFL", "FUFR", "FDFL", "FDFR"}
STANCE_TOKEN = re.compile(r"^[A-Z]{2,5}$")
GENERATED_MARK = "generated skeleton"


def stance_names(source: dict) -> list[str]:
    names: list[str] = []

    def add(token: str) -> None:
        upper = token.strip().upper()
        if STANCE_TOKEN.fullmatch(upper) and upper not in ENGINE_STATES | GROUND_POSITIONS | {"WHILE"} \
                and upper not in names:
            names.append(upper)

    for token in source.get("stances") or []:
        add(str(token))
    for move in source.get("moves") or []:
        recovery = str(move.get("recovery") or "")
        ending = re.search(r"(?:^|\s)(?:r\d+(?:~\d+)?\s+)?([A-Za-z]{2,5})\s*$", recovery)
        if ending:
            add(ending.group(1))
        for prefix in re.findall(r"(?:^|,)\s*([A-Za-z]{2,5})\.", str(move.get("command") or "")):
            if prefix.upper() in {name.upper() for name in source.get("stances") or []}:
                add(prefix)
    return names


def skeleton(slug: str, names: list[str]) -> str:
    document = {
        "schema_version": 1,
        "character": slug,
        "evidence": [f"data/characters/{slug}.yaml stance list and move notes ({GENERATED_MARK}, "
                     "tools/generate_state_machines.py)"],
        "stances": {name: {"can_guard": False, "max_frames": None,
                           "status": {"can_guard": "provisional", "max_frames": "provisional"}}
                    for name in names},
    }
    header = (f"# {GENERATED_MARK} by tools/generate_state_machines.py: every value is a provisional\n"
              "# stand-in (cannot guard, no time limit), as for Jun's undocumented stances. Replace\n"
              "# them with guide or Practice-mode evidence and change the status; a reviewed file is\n"
              "# never regenerated. Add parry outcomes, pulses, and a resource gauge here too.\n")
    return header + yaml.safe_dump(document, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    roster = yaml.safe_load((args.data_root / "roster.yaml").read_text(encoding="utf-8"))
    changed: list[str] = []
    for row in roster["characters"]:
        slug = str(row["slug"])
        source_path = args.data_root / "characters" / f"{slug}.yaml"
        target = args.data_root / "character_modules" / slug / "state_machine.yaml"
        if not source_path.exists():
            continue
        if target.exists() and GENERATED_MARK not in target.read_text(encoding="utf-8").splitlines()[0]:
            continue  # reviewed by a person
        names = stance_names(yaml.safe_load(source_path.read_text(encoding="utf-8")) or {})
        if not names:
            continue
        text = skeleton(slug, names)
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            changed.append(slug)
            if not args.check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
    print(f"{'would change' if args.check else 'wrote'} {len(changed)} state machines: {', '.join(changed)}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())

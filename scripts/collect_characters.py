"""Overnight collection: switches characters in Practice mode and, for each, validates its moves
in the game and records their pose sequences (the opponent-move recognizer's training data).

    python scripts/collect_characters.py [--characters jin,king,...] [--reps 1] [--delay 5]

Offline Practice mode, your own controller unplugged; leave the game alone (Esc stops). One
virtual pad is used for the whole run (a new pad would bring up "Press Menu" again). For every
character still to do:

  1. Start menu -> Character Selection (found by where the highlighted row is) -> Yes;
  2. P1 = the character and P2 = Azucena, chosen by reading the hovered names
     (t8_agent.live.menu_automation); the stage right of Random; wait for Practice to load;
  3. identity check: the character's jab (command "1") must show its startup and damage;
  4. each move testable from a standing start is performed --reps times while both skeletons
     are recorded; the Practice panel then shows its startup and damage. A move that matches
     our data passes: its recording is kept as recognizer data, and it is recorded in
     data/validation/<character>.json (method practice_mode, reviewer auto_validator). Then
     the non-attacks (idle, walks, crouch, dashes).

Progress is kept in runs/collect_characters/progress.json, so a stopped run resumes where it left
off; a character that fails three times is skipped. The log is runs/collect_characters.log.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_calibrate_distance import fresh_frame, hold, readable  # noqa: E402
from record_move_dataset import NON_ATTACKS, record  # noqa: E402
from validate_moves import testable  # noqa: E402
from t8_agent.io.pose_tracker import PoseFighterTracker  # noqa: E402
from t8_agent.live.menu_automation import CharacterSelect, pad_presser  # noqa: E402
from t8_agent.live.practice_readout import PracticeDistanceReader, PracticeLineReader, panel_crop  # noqa: E402

ORDER = ("jin kazuya king paul law jack-8 lars xiaoyu nina leroy asuka lili bryan hwoarang claudio azucena "
         "raven leo steve kuma yoshimitsu shaheen dragunov feng panda lee alisa zafina devil-jin victor reina eddy "
         "lidia heihachi clive anna fahkumram armor-king miary-zo kunimitsu").split()
MENU_CHARACTER_SELECTION_Y = 654  # full-resolution y of the "Character Selection" row in the pause menu
MENU_ROW_HEIGHT = 45


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open("runs/collect_characters.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def highlighted_menu_row(frame: np.ndarray) -> int | None:
    """y (full resolution) of the pause menu's highlighted row, or None when no menu is open.

    The highlight is a red bar across the menu column (a bright band sweeps along it, so its
    colour runs from (77,36,53) to (189,39,71)); the menu around it is dark blue-grey (18,27,39).
    A row band counts when it is 25-60 px tall, red across most of the column, and the menu's dark
    background lies just above or below it (which no fight screen or dialog has there)."""
    scale = frame.shape[1] / 1920.0
    left, right = int(95 * scale), int(550 * scale)
    column = frame[:, left:right, :3].astype(int)
    red, green, blue = column[..., 0], column[..., 1], column[..., 2]
    pink = (red > 60) & (red > green + 25) & (red > blue + 12)
    full = pink.mean(axis=1) > 0.5
    bands, start = [], None
    for y, on in enumerate(list(full) + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append((start, y - 1))
            start = None
    brightness = column.max(axis=2)
    # A menu is open only if its dark panel fills the column (180-900 px): fight screens are bright there.
    panel = brightness[int(180 * scale):int(900 * scale)]
    if np.median(panel) > 70:
        return None
    for top, bottom in bands:
        height = (bottom - top + 1) / scale
        if not 25 <= height <= 60:
            continue
        gap = int(12 * scale)
        above = brightness[max(0, top - 3 * gap):max(1, top - gap)]
        below = brightness[bottom + gap:bottom + 3 * gap]
        dark = [part.size and np.median(part) < 60 for part in (above, below)]
        if any(dark):  # the menu's first or last row has the menu's edge on one side
            return int((top + bottom) / 2 / scale)
    return None


def settled_reading(reads: list[dict]) -> dict:
    """The panel values to judge a move by, from per-frame reads: Practice clears the startup and
    frame advantage lines about 2.2 s after a move, so the newest pair of agreeing consecutive
    reads that still shows a startup is taken (else the newest agreeing pair)."""
    pairs = [a for a, b in zip(reads, reads[1:]) if a == b]
    with_startup = [a for a in pairs if a.get("startup") is not None]
    if with_startup:
        return with_startup[-1]
    return pairs[-1] if pairs else {}


def dialog_highlight(frame: np.ndarray) -> str | None:
    """A centred Yes/No dialog ("Return to Side Selection?", "Return to Character Select?"): which
    answer is highlighted ("yes" above, "no" below, a ~40 px red bar across the dialog), or None."""
    scale = frame.shape[1] / 1920.0
    top = int(520 * scale)
    box = frame[top:int(680 * scale), int(560 * scale):int(1360 * scale), :3].astype(int)
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    rows = np.nonzero(((r > 85) & (r > g + 50) & (r > b + 35)).mean(axis=1) > 0.8)[0]
    if not 25 * scale <= len(rows) <= 55 * scale:
        return None
    centre = (top + rows.mean()) / scale
    return "no" if centre > 594 else "yes"  # No's bar is centred near y=616, Yes's near 572


def stage_hovered(frame: np.ndarray) -> str | None:
    """On the stage select screen: "random" or "arena" (the stage right of it) when that tile is
    highlighted, else None (also on every other screen). A hovered tile has a red frame; Arena's
    also glows pink inside (45% red pixels vs Random's 10%)."""
    scale = frame.shape[1] / 1920.0

    def red_share(x0: int, x1: int) -> float:
        tile = frame[int(700 * scale):int(804 * scale), int(x0 * scale):int(x1 * scale), :3].astype(int)
        r, g, b = tile[..., 0], tile[..., 1], tile[..., 2]
        return float(((r > 150) & (r > g + 80) & (r > b + 40)).mean())

    random_tile, arena_tile = red_share(250, 410), red_share(424, 584)
    if arena_tile > 0.25:
        return "arena"
    if random_tile > 0.06 and arena_tile < 0.1:
        return "random"
    return None


class Collector:
    def __init__(self, args) -> None:
        import dxcam
        import keyboard
        from t8_agent.io.controller_backend import VGamepadInputBackend
        from t8_agent.moves.catalog import load_compiled_catalog
        self.args = args
        self.keyboard = keyboard
        self.camera = dxcam.create(output_idx=0)
        self.distance = PracticeDistanceReader()
        self.lines = PracticeLineReader()
        self.pad = VGamepadInputBackend(
            facing=1, tap_seconds=0.035, asynchronous=True, hold_timeout=0.25, string_gap_seconds=14 / 60,
            string_first_gap_seconds=8 / 60, string_gap_growth_seconds=4 / 60, dash_button_delay_seconds=4 / 60)
        self.press = pad_presser(self.pad)
        self.grab = lambda: fresh_frame(self.camera, 1.0)
        self.select = CharacterSelect(self.camera, self.press, self.grab)
        config = yaml.safe_load(Path("config/live_screen.yaml").read_text(encoding="utf-8")) or {}
        self.tracker = PoseFighterTracker(config)
        self.tracker.update(np.zeros((1080, 1920, 3), np.uint8))  # loads the pose model (seconds) now,
        self.tracker.reset_round()                                # not during the first recording
        self.catalog = load_compiled_catalog(Path("data/generated/full_move_catalog.json"))
        self.moves = {move["stable_id"]: move for move in self.catalog.moves}

    def stop(self) -> bool:
        return self.keyboard.is_pressed("esc")

    def in_fight(self) -> bool:
        return readable(self.camera, self.distance) is not None

    # ---------------------------------------------------------------- menus
    def where(self) -> str:
        """"fight", "menu" (the pause menu), "select_p1" / "select_p2" (character select, whose
        cursor is active: once P1 is picked both names show), or "other" (the "Press Menu"
        prompt, a dialog, loading)."""
        if self.in_fight():
            return "fight"
        frame = self.grab()
        if frame is not None and highlighted_menu_row(frame) is not None:
            return "menu"
        p1 = self.select.hovered(1)[1] < 0.15
        p2 = self.select.hovered(2)[1] < 0.15
        if p1 and p2:
            return "select_p2"
        if p1:
            return "select_p1"
        return "other"

    def snapshot(self, tag: str) -> None:
        """Saves the screen for the morning's diagnosis."""
        import cv2
        frame = self.grab()
        if frame is not None:
            cv2.imwrite(f"runs/collect_characters/{time.strftime('%H%M%S')}_{tag}.png",
                        cv2.resize(frame, (960, 540))[:, :, ::-1])

    def choose_menu_row(self) -> bool:
        """In the pause menu: moves to "Character Selection" and confirms it (and the dialog)."""
        for _ in range(20):
            row = None
            for _ in range(3):  # the bar pulses: try a few frames
                frame = self.grab()
                row = highlighted_menu_row(frame) if frame is not None else None
                if row is not None:
                    break
                time.sleep(0.1)
            if row is None:
                return False
            if abs(row - MENU_CHARACTER_SELECTION_Y) < MENU_ROW_HEIGHT // 2:
                break
            self.press("down" if row < MENU_CHARACTER_SELECTION_Y else "up", after=0.2)
        else:
            return False
        self.press("a", after=1.0)    # "Return to Character Select?" (No highlighted)
        self.press("up", after=0.4)   # Yes
        self.press("a", after=4.0)
        return True

    def to_character_select(self) -> bool:
        """From any screen to character select with the P1 cursor active."""
        others = 0
        for _ in range(24):
            state = self.where()
            if state == "select_p1":
                return True
            if state == "select_p2":
                self.press("b", after=1.5)  # back to P1's choice
            elif state == "fight":
                self.press("start", after=1.2)  # the pause menu
            elif state == "menu":
                if not self.choose_menu_row():
                    self.press("b", after=1.0)
            else:
                frame = self.grab()
                dialog = dialog_highlight(frame) if frame is not None else None
                stage = stage_hovered(frame) if frame is not None else None
                if dialog == "no":  # only A works in these dialogs: keep the current screen
                    self.press("a", after=1.5)
                elif dialog == "yes":
                    self.press("down", after=0.4)
                elif stage == "random":  # stage select: go on into the fight (then the pause menu)
                    self.press("right", after=0.6)
                elif stage == "arena":
                    self.press("a", after=1.0)
                    for _ in range(40):  # loading
                        if self.in_fight():
                            break
                        time.sleep(1.0)
                else:
                    # The "Press Menu" prompt takes Start; loading screens just need time.
                    others += 1
                    self.press("start" if others % 2 else "b", after=2.0)
        self.snapshot("no_character_select")
        return False

    def load(self, character: str) -> bool:
        if not self.to_character_select():
            log(f"{character}: could not reach character select")
            return False
        if not self.select.select(character, 1):
            log(f"{character}: not found on the grid")
            self.snapshot(f"{character}_not_found")
            return False
        time.sleep(2.0)
        if not self.select.select("azucena", 2):
            log(f"{character}: could not pick the P2 dummy")
            return False
        # The stage select appears after a few seconds; pick Arena, right of Random.
        for _ in range(60):
            frame = self.grab()
            hovered = stage_hovered(frame) if frame is not None else None
            if hovered == "random":
                self.press("right", after=0.6)
            elif hovered == "arena":
                self.press("a", after=1.0)
                break
            time.sleep(0.5)
        else:
            log(f"{character}: no stage select")
            self.snapshot(f"{character}_no_stage")
            return False
        for _ in range(60):
            if self.in_fight():
                return True
            time.sleep(1.0)
        log(f"{character}: Practice did not load")
        self.snapshot(f"{character}_no_practice")
        return False

    # ---------------------------------------------------------------- moves
    def perform(self, table, index: int) -> tuple[dict, dict]:
        """Restart, step in, perform the move while recording; then read the panel."""
        self.pad.press_practice_restart()
        time.sleep(2.2)
        self.tracker.reset_round()
        hold(self.pad, "dash_forward", 0.15, self.stop)
        time.sleep(0.35)
        self.pad.set_enabled(True)
        sid = table.stable_ids[index]
        reads: list[dict] = []

        def read(frame: np.ndarray) -> None:
            panel = panel_crop(frame[:, :, ::-1].copy())
            reads.append({line: self.lines.read(panel, line)
                          for line in ("total_damage", "damage", "startup", "frame_advantage")})

        sample = record(self.camera, self.tracker, self.pad, 2.2,
                        lambda: self.pad.send_command(table.commands[index], move_id=sid), self.stop, on_frame=read)
        self.pad.release_all()
        return sample, settled_reading(reads)

    def judge(self, move: dict, reading: dict) -> str:
        damages = [float(value) for value in (move.get("damage") or [])]
        shown = reading.get("damage")
        reached = max((k for k, value in enumerate(damages) if shown is not None and abs(value - shown) < 0.5),
                      default=None)
        startup = move.get("startup") or {}
        if reached is None:
            return "whiff"
        if len(damages) <= 1:
            measured = reading.get("startup")
            low, high = startup.get("min"), startup.get("max") or startup.get("min")
            ok = measured is not None and low is not None and low - 1 <= measured <= high + 1
            return "pass" if ok else "fail"
        return "pass" if reached == len(damages) - 1 else f"partial {reached + 1}/{len(damages)}"

    def collect(self, character: str) -> dict:
        from t8_agent.moves.candidates import build_candidate_table
        table = build_candidate_table(self.catalog, character)
        indices = [i for i, command in enumerate(table.commands)
                   if not table.is_universal(i) and command is not None and testable(command)
                   and (self.moves.get(table.stable_ids[i]) or self.moves.get(table.stable_ids[i].split("~")[0]))]
        # Identity: the jab first.
        jab = next((i for i in indices if (self.moves.get(table.stable_ids[i]) or {}).get("command", {}).get("raw") == "1"), None)
        if jab is not None:
            _, reading = self.perform(table, jab)
            verdict = self.judge(self.moves[table.stable_ids[jab]], reading)
            if verdict != "pass":
                _, reading = self.perform(table, jab)
                verdict = self.judge(self.moves[table.stable_ids[jab]], reading)
            if verdict != "pass":
                log(f"{character}: jab check failed ({reading}); wrong character loaded?")
                return {"status": "identity failed"}
        out = Path("runs/move_dataset") / character
        out.mkdir(parents=True, exist_ok=True)
        ledger_path = Path("data/validation") / f"{character}.json"
        ledger = (json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists()
                  else {"schema_version": 1, "moves": {}, "scenarios": {}, "routes": {}})
        provenance = {"patch": self.args.patch, "recorded_at": date.today().isoformat(), "method": "practice_mode",
                      "confidence": "medium", "reviewer": "auto_validator", "review_status": "unreviewed"}
        counts = {"pass": 0, "other": 0}
        for rep in range(self.args.reps):
            for done, index in enumerate(indices):
                if self.stop():
                    raise KeyboardInterrupt
                if done and done % 25 == 0:
                    log(f"{character}: {done}/{len(indices)} moves, {counts['pass']} passed")
                sid = table.stable_ids[index]
                move = self.moves.get(sid) or self.moves.get(sid.split("~")[0])
                try:
                    sample, reading = self.perform(table, index)
                except ValueError as error:  # the controller cannot send it (a state requirement)
                    self.pad.release_all()
                    log(f"{character}: {sid} skipped ({error})")
                    counts["other"] += 1
                    continue
                verdict = self.judge(move, reading)
                if verdict != "pass":
                    counts["other"] += 1
                    continue
                counts["pass"] += 1
                ledger["moves"][sid] = {"status": "pass", "measured_startup": reading.get("startup"),
                                        "measured_damage": reading.get("damage"),
                                        "measured_hit_advantage": reading.get("frame_advantage"),
                                        "provenance": provenance}
                if len(sample["times"]) >= 10:
                    labels = {"stable_id": sid, "attack": True, "hit_levels": move.get("hit_levels") or [],
                              "startup": (move.get("startup") or {}).get("min"), "damage": move.get("damage") or [],
                              "measured_startup": reading.get("startup")}
                    np.savez_compressed(out / f"{sid.replace(':', '_').replace('~', '-')}_{rep}.npz",
                                        times=sample["times"], p1=sample["p1"], p2=sample["p2"], boxes=sample["boxes"],
                                        sent_at=np.float32(sample["sent_at"]), labels=json.dumps(labels))
            ledger_path.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            for name in NON_ATTACKS:
                self.pad.press_practice_restart()
                time.sleep(2.2)
                self.tracker.reset_round()
                hold(self.pad, "dash_forward", 0.15, self.stop)
                time.sleep(0.35)
                self.pad.set_enabled(True)
                sample = record(self.camera, self.tracker, self.pad, 2.0, lambda n=name: self.pad.send_action(n),
                                self.stop, held=name in ("walk_forward", "walk_back", "crouch"))
                self.pad.release_all()
                if len(sample["times"]) >= 10:
                    labels = {"stable_id": f"universal:{name}", "attack": False, "hit_levels": [], "startup": None,
                              "damage": []}
                    np.savez_compressed(out / f"universal_{name}_{rep}.npz", times=sample["times"], p1=sample["p1"],
                                        p2=sample["p2"], boxes=sample["boxes"], sent_at=np.float32(sample["sent_at"]),
                                        labels=json.dumps(labels))
        return {"status": "done", "moves": len(indices), "passed": counts["pass"], "not_passed": counts["other"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--characters", default=",".join(ORDER))
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--delay", type=float, default=8, help="seconds for the new pad to register")
    parser.add_argument("--patch", default="3.02.02")
    args = parser.parse_args()
    Path("runs/collect_characters").mkdir(parents=True, exist_ok=True)
    progress_path = Path("runs/collect_characters/progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    collector = Collector(args)
    time.sleep(args.delay)
    collector.press("start", after=2.0)  # answers the "Press Menu" prompt a new pad brings up
    log(f"starting on screen: {collector.where()}")
    try:
        for character in args.characters.split(","):
            state = progress.get(character, {})
            if state.get("status") == "done" or state.get("failures", 0) >= 3:
                continue
            log(f"{character}: loading")
            if not collector.load(character):
                state["failures"] = state.get("failures", 0) + 1
                progress[character] = state
                progress_path.write_text(json.dumps(progress, indent=1), encoding="utf-8")
                continue
            log(f"{character}: collecting")
            try:
                result = collector.collect(character)
            except KeyboardInterrupt:
                raise
            except Exception as error:  # one character's failure must not end the night
                import traceback
                log(f"{character}: crashed: {error!r} " + traceback.format_exc().replace(chr(10), " | "))
                collector.pad.release_all()
                result = {"status": "crashed"}
            if result["status"] != "done":
                result["failures"] = state.get("failures", 0) + 1
            progress[character] = result
            progress_path.write_text(json.dumps(progress, indent=1), encoding="utf-8")
            log(f"{character}: {result}")
    except KeyboardInterrupt:
        log("stopped (Esc)")
    finally:
        collector.pad.release_all()
        collector.pad.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

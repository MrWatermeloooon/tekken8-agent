from __future__ import annotations

from dataclasses import asdict, dataclass
import re


_DIRECTIONS = {
    "n", "f", "b", "u", "d", "uf", "ub", "df", "db",
    "fF", "bB", "uU", "dD", "qcf", "qcb", "hcf", "hcb",
}
_BUTTON_RE = re.compile(r"^(?:[1-4](?:\+[1-4])*)$")
_FRAME_RE = re.compile(r"(?P<minimum>\d+)(?:~(?P<maximum>\d+))?[fF]?$")
_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 +_-]*$")
_TOKEN_CLEAN_RE = re.compile(r"\s+")

# Situational requirements: they gate when a row can occur and are never
# button inputs themselves.
BACK_TO_WALL = "BACK_TO_WALL"
OPPONENT_BACK_TURNED = "OPPONENT_BACK_TURNED"
OPPONENT_LEFT_SIDE = "OPPONENT_LEFT_SIDE"
OPPONENT_RIGHT_SIDE = "OPPONENT_RIGHT_SIDE"
PARRY_SUCCESS = "PARRY_SUCCESS"
SITUATIONAL_REQUIREMENTS = frozenset({
    BACK_TO_WALL, OPPONENT_BACK_TURNED, OPPONENT_LEFT_SIDE, OPPONENT_RIGHT_SIDE,
    PARRY_SUCCESS, f"{PARRY_SUCCESS}_HIGH", f"{PARRY_SUCCESS}_MID", f"{PARRY_SUCCESS}_LOW",
    f"{PARRY_SUCCESS}_THROW",
})

# "Back throw" / "Left throw" / "Right throw": any throw input (1+3 or 2+4)
# performed with the opponent's back or that side facing the player.
_POSITIONAL_THROW_RE = re.compile(r"^(?P<side>back|left|right)\s+throw$", re.IGNORECASE)
_POSITIONAL_THROWS = {
    "back": OPPONENT_BACK_TURNED, "left": OPPONENT_LEFT_SIDE, "right": OPPONENT_RIGHT_SIDE,
}
# Leading parenthesized situations with an exact, documented meaning.
_SITUATION_PREFIX_RE = re.compile(r"^\((?P<situation>[^)]+)\)\s*\.?\s*(?P<rest>.+)$")
_SITUATION_PREFIXES = {"back to wall": BACK_TO_WALL}
# TekkenDocs writes a successful parry/absorb as a "P" step, optionally with
# the parried type: "b+1+3,P", "GEN.P (Low)". It is an event, not the punch
# button; every roster row using it documents a parry or absorb.
_PARRY_EVENT_RE = re.compile(r"^P(?:\s*\((?P<kind>high|mid|low|throw)\))?$", re.IGNORECASE)


@dataclass(frozen=True)
class InputStep:
    raw: str
    direction: str = "n"
    buttons: tuple[int, ...] = ()
    hold: bool = False
    just_frame: bool = False
    delay_min: int = 0
    delay_max: int = 0
    release: bool = False
    requirement: str | None = None
    uncertain: bool = False

    def to_dict(self) -> dict:
        value = asdict(self)
        value["buttons"] = list(self.buttons)
        return value


@dataclass(frozen=True)
class CommandSpec:
    raw: str
    requirements: tuple[str, ...]
    steps: tuple[InputStep, ...]
    parser_status: str
    warnings: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.parser_status == "parsed" and bool(self.steps)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "requirements": list(self.requirements),
            "steps": [step.to_dict() for step in self.steps],
            "parser_status": self.parser_status,
            "warnings": list(self.warnings),
        }


def parse_command(command: str, known_stances: set[str] | None = None) -> CommandSpec:
    """Parse TekkenDocs command notation without pretending ambiguous text is exact.

    The parser recognizes ordinary directions, button chords, strings, holds,
    release inputs, just-frame separators, and stance/state prefixes. Unknown
    fragments remain in the result with ``uncertain=True`` and make the command
    non-executable until its per-character validation record resolves them.
    """

    raw = str(command or "").strip()
    if not raw:
        return CommandSpec(raw, (), (), "invalid", ("empty command",))

    throw = _POSITIONAL_THROW_RE.fullmatch(raw)
    if throw:
        return CommandSpec(
            raw, (_POSITIONAL_THROWS[throw.group("side").lower()],),
            (InputStep(raw="1+3", buttons=(1, 3)),), "parsed")

    known = {value.upper() for value in (known_stances or set())}
    requirements: list[str] = []
    situation = _SITUATION_PREFIX_RE.fullmatch(raw)
    if situation and situation.group("situation").strip().lower() in _SITUATION_PREFIXES:
        requirements.append(_SITUATION_PREFIXES[situation.group("situation").strip().lower()])
        raw_body = situation.group("rest").strip()
    else:
        raw_body = raw
    body = raw_body
    if "." in raw_body:
        prefix, body = raw_body.rsplit(".", 1)
        prefix_parts = [value.strip() for value in prefix.split(".") if value.strip()]
        for value in prefix_parts:
            upper = value.upper()
            if upper in known or upper in {"H", "R", "WS", "FC", "SS", "BT", "WR", "HFC"}:
                requirements.append(upper)
            elif _PREFIX_RE.fullmatch(value):
                requirements.append(upper)
            else:
                return CommandSpec(raw, tuple(requirements), (), "invalid", (f"invalid prefix: {value}",))

    fragments = _split_sequence(body)
    steps: list[InputStep] = []
    warnings: list[str] = []
    for fragment in fragments:
        for expanded, just_frame in _expand_transitions(fragment):
            step, warning = _parse_step(expanded, just_frame)
            steps.append(step)
            if warning:
                warnings.append(warning)
    status = "parsed" if steps and not any(step.uncertain for step in steps) else "needs_review"
    return CommandSpec(raw, tuple(requirements), tuple(steps), status, tuple(warnings))


def _split_sequence(value: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for character in value:
        if character in "([{" :
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        if character == "," and depth == 0:
            fragment = "".join(current).strip()
            if fragment:
                result.append(fragment)
            current.clear()
        else:
            current.append(character)
    fragment = "".join(current).strip()
    if fragment:
        result.append(fragment)
    return result


def _expand_transitions(fragment: str) -> list[tuple[str, bool]]:
    """Split sequential-button and just-frame separators, preserving chords."""

    result: list[tuple[str, bool]] = []
    current: list[str] = []
    depth = 0
    next_is_just_frame = False
    for character in fragment:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        if character in "~:" and depth == 0:
            value = "".join(current).strip()
            if value:
                result.append((value, next_is_just_frame))
            current.clear()
            next_is_just_frame = character == ":"
        else:
            current.append(character)
    value = "".join(current).strip()
    if value:
        result.append((value, next_is_just_frame))
    return result or [(fragment, False)]


def _parse_step(fragment: str, just_frame: bool = False) -> tuple[InputStep, str | None]:
    raw = fragment.strip()
    parry = _PARRY_EVENT_RE.fullmatch(raw)
    if parry:
        kind = parry.group("kind")
        return InputStep(raw=raw, requirement=PARRY_SUCCESS + (f"_{kind.upper()}" if kind else "")), None
    token = _TOKEN_CLEAN_RE.sub("", raw)
    just_frame = just_frame or "!" in token
    release = token.startswith("[") and token.endswith("]")
    if release:
        token = token[1:-1]
    token = token.replace("!", "")

    requirement = None
    state_match = re.match(r"^(ws|ss|bt|fc|wr|ch)(?=[udfbnqhc1-4])", token, re.IGNORECASE)
    if state_match:
        requirement = state_match.group(1).upper()
        token = token[state_match.end():]

    delay_min = delay_max = 0
    delay_match = re.search(r"(?:~|\{)(\d+)(?:~(\d+))?[fF]?\}?$", token)
    if delay_match:
        delay_min = int(delay_match.group(1))
        delay_max = int(delay_match.group(2) or delay_match.group(1))
        token = token[: delay_match.start()]

    hold = any(character.isupper() for character in token if character.isalpha())
    parts = [part for part in token.split("+") if part]
    direction = "n"
    button_parts: list[str] = []
    uncertain = False
    for part in parts:
        normalized = part.replace("/", "").replace("*", "")
        direction_candidate = normalized.lower()
        if direction_candidate in {value.lower() for value in _DIRECTIONS}:
            direction = normalized
        elif _BUTTON_RE.fullmatch(normalized):
            button_parts.extend(normalized.split("+"))
        elif normalized in {"N", "n"}:
            direction = "n"
        else:
            uncertain = True
    if not parts:
        uncertain = True
    buttons = tuple(sorted({int(value) for value in button_parts}))
    if direction == "n" and not buttons and token.isdigit() and token in {"1", "2", "3", "4"}:
        buttons = (int(token),)
        uncertain = False
    warning = f"unparsed input fragment: {raw}" if uncertain else None
    return InputStep(
        raw=raw,
        direction=direction,
        buttons=buttons,
        hold=hold,
        just_frame=just_frame,
        delay_min=delay_min,
        delay_max=delay_max,
        release=release,
        requirement=requirement,
        uncertain=uncertain,
    ), warning

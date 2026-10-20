from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import html
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
OPPONENT_WALL_STUN = "OPPONENT_WALL_STUN"
OPPONENT_WALL_SPLAT = "OPPONENT_WALL_SPLAT"
GROUNDED_FACE_UP = "GROUNDED_FACE_UP"
AIRBORNE = "AIRBORNE"
LANDING = "LANDING"
COUNTER_HIT = "COUNTER_HIT"
RUN_TWO_STEPS = "RUN_TWO_STEPS"
SITUATIONAL_REQUIREMENTS = frozenset({
    BACK_TO_WALL, OPPONENT_BACK_TURNED, OPPONENT_LEFT_SIDE, OPPONENT_RIGHT_SIDE,
    PARRY_SUCCESS, f"{PARRY_SUCCESS}_HIGH", f"{PARRY_SUCCESS}_MID", f"{PARRY_SUCCESS}_LOW",
    f"{PARRY_SUCCESS}_THROW", OPPONENT_WALL_STUN, OPPONENT_WALL_SPLAT, GROUNDED_FACE_UP,
    AIRBORNE, LANDING, COUNTER_HIT, RUN_TWO_STEPS,
})

# "Back throw", "(Left Side Throw)", "(Behind 2 throw)", "Back Throw 2+4": a
# throw performed with the opponent's back or that side facing the player.
# Without a documented button it is any throw input, written as 1+3; "Behind
# 1/2" and an explicit trailing chord name the throw input exactly.
_POSITIONAL_THROW_RE = re.compile(
    r"^\(?(?P<side>back|behind|left|right)(?:\s+side)?\s+(?:(?P<button>[12])\s+)?throw\)?"
    r"(?:\s+(?P<chord>1\+3|2\+4))?$", re.IGNORECASE)
_POSITIONAL_THROWS = {
    "back": OPPONENT_BACK_TURNED, "behind": OPPONENT_BACK_TURNED,
    "left": OPPONENT_LEFT_SIDE, "right": OPPONENT_RIGHT_SIDE,
}
# Leading situations with an exact, documented meaning, parenthesized or as a
# plain "While ..." phrase ("While landing 3").
_SITUATION_PREFIX_RE = re.compile(r"^\((?P<situation>[^)]+)\)\s*\.?\s*(?P<rest>.+)$")
_WHILE_PREFIX_RE = re.compile(r"^(?P<situation>while (?:in the air|landing))\s+(?P<rest>.+)$",
                              re.IGNORECASE)
_SITUATION_PREFIXES = {
    "back to wall": BACK_TO_WALL,
    "during enemy wall stun": OPPONENT_WALL_STUN,
    "wall splat": OPPONENT_WALL_SPLAT,
    "while down, facing up": GROUNDED_FACE_UP,
    "while in the air": AIRBORNE,
    "while landing": LANDING,
    "2 steps or more": RUN_TWO_STEPS,
}
# Trailing outcome qualifiers: "d+4 (counterhit)" applies to the whole
# command, "d+3,1(wall stun)" (no space) to the step it is attached to.
_SITUATION_SUFFIX_RE = re.compile(r"^(?P<rest>.+?)(?P<space>\s*)\((?P<situation>counterhit|wall stun)\)$",
                                  re.IGNORECASE)
_SITUATION_SUFFIXES = {"counterhit": COUNTER_HIT, "wall stun": OPPONENT_WALL_STUN}
# Documented alternative inputs for one row ("AIR.1+3 / AIR.f+1+3"): the
# first is the canonical command, the others are kept for validation.
_ALTERNATIVE_SPLIT_RE = re.compile(r"\s*/\s+|\s+/\s*")
# A stance prefix with a documented qualifier: "CLK(Two spins)".
_QUALIFIED_PREFIX_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9_-]*)\((?P<qualifier>[A-Za-z0-9 ]+)\)$")
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
    alternatives: tuple[str, ...] = ()

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
            "alternatives": list(self.alternatives),
        }


def parse_command(command: str, known_stances: set[str] | None = None) -> CommandSpec:
    """Parse TekkenDocs command notation without pretending ambiguous text is exact.

    The parser recognizes ordinary directions, button chords, strings, holds,
    release inputs, just-frame separators, and stance/state prefixes. Unknown
    fragments remain in the result with ``uncertain=True`` and make the command
    non-executable until its per-character validation record resolves them.
    """

    raw = html.unescape(str(command or "")).strip()
    if not raw:
        return CommandSpec(raw, (), (), "invalid", ("empty command",))

    alternatives = [value.strip() for value in _ALTERNATIVE_SPLIT_RE.split(raw) if value.strip()]
    if len(alternatives) > 1:
        primary = parse_command(alternatives[0], known_stances)
        status = primary.parser_status
        if status == "parsed" and any(
                parse_command(value, known_stances).parser_status != "parsed" for value in alternatives[1:]):
            status = "needs_review"
        return CommandSpec(
            raw, primary.requirements, primary.steps, status,
            primary.warnings + (f"alternative inputs: {' / '.join(alternatives[1:])}",),
            tuple(alternatives[1:]))

    throw = _POSITIONAL_THROW_RE.fullmatch(raw)
    if throw:
        chord = throw.group("chord") or ("2+4" if throw.group("button") == "2" else "1+3")
        return CommandSpec(
            raw, (_POSITIONAL_THROWS[throw.group("side").lower()],),
            (InputStep(raw=chord, buttons=tuple(int(value) for value in chord.split("+"))),), "parsed")

    known = {value.upper() for value in (known_stances or set())}
    requirements: list[str] = []
    raw_body = raw
    while True:
        situation = _SITUATION_PREFIX_RE.fullmatch(raw_body) or _WHILE_PREFIX_RE.fullmatch(raw_body)
        key = situation.group("situation").strip().lower() if situation else ""
        if key not in _SITUATION_PREFIXES:
            break
        requirements.append(_SITUATION_PREFIXES[key])
        raw_body = situation.group("rest").strip()
    step_suffix = None
    suffix = _SITUATION_SUFFIX_RE.fullmatch(raw_body)
    if suffix:
        raw_body = suffix.group("rest").strip()
        situation_name = _SITUATION_SUFFIXES[suffix.group("situation").lower()]
        if suffix.group("space"):
            requirements.append(situation_name)
        else:
            step_suffix = situation_name

    fragments = _split_sequence(raw_body)
    if fragments and "." in fragments[0] and not re.search(r"[~:]", fragments[0].rsplit(".", 1)[0]):
        prefix, fragments[0] = fragments[0].rsplit(".", 1)
        for value in (part.strip() for part in prefix.split(".") if part.strip()):
            requirement = _prefix_requirement(value, known)
            if requirement is None:
                return CommandSpec(raw, tuple(requirements), (), "invalid", (f"invalid prefix: {value}",))
            requirements.append(requirement)
    elif len(fragments) > 1 and fragments[0].upper() in known and _PREFIX_RE.fullmatch(fragments[0]):
        # A bare state name opening a string ("UT,1+2") is that state's prefix.
        requirements.append(fragments.pop(0).upper())

    steps: list[InputStep] = []
    warnings: list[str] = []
    for fragment in fragments:
        # A stance written at the start of a fragment ("NSS.1~1") holds for
        # each of that fragment's transitions.
        fragment_requirement = None
        for position, (expanded, just_frame) in enumerate(_expand_transitions(fragment)):
            step_requirement = fragment_requirement
            if steps:
                expanded, prefix_requirement, event = _split_step_prefix(expanded, known)
                if event is not None:
                    steps.append(event)
                if prefix_requirement:
                    step_requirement = prefix_requirement
                    if position == 0:
                        fragment_requirement = prefix_requirement
                if expanded is None:
                    continue
            step, warning = _parse_step(expanded, just_frame)
            if step_requirement:
                if step.requirement:
                    step = replace(step, uncertain=True)
                    warning = warning or f"conflicting step requirements: {expanded}"
                else:
                    step = replace(step, requirement=step_requirement)
            steps.append(step)
            if warning:
                warnings.append(warning)
    if step_suffix and steps:
        if steps[-1].requirement:
            steps[-1] = replace(steps[-1], uncertain=True)
            warnings.append(f"conflicting step requirements: {steps[-1].raw}")
        else:
            steps[-1] = replace(steps[-1], requirement=step_suffix)
    status = "parsed" if steps and not any(step.uncertain for step in steps) else "needs_review"
    return CommandSpec(raw, tuple(requirements), tuple(steps), status, tuple(warnings))


def _prefix_requirement(value: str, known: set[str]) -> str | None:
    upper = value.upper()
    if upper in known or upper in {"H", "R", "WS", "FC", "SS", "BT", "WR", "HFC"}:
        return upper
    if _PREFIX_RE.fullmatch(value):
        return upper
    qualified = _QUALIFIED_PREFIX_RE.fullmatch(value)
    if qualified:
        return f"{qualified.group('name').upper()}({qualified.group('qualifier').strip().upper()})"
    return None


def _split_step_prefix(
    fragment: str, known: set[str],
) -> tuple[str | None, str | None, InputStep | None]:
    """Split a later step's "STANCE.input" or "P.input" into its parts.

    TekkenDocs writes the stance a later hit is performed in ("1,NSS.1") and a
    parry outcome followed by its follow-up ("b+1+3,P.2") this way. Returns the
    remaining input, the step requirement, and a parry event step, if any.
    """

    if "." not in fragment:
        return fragment, None, None
    prefix, rest = (value.strip() for value in fragment.split(".", 1))
    parry = _PARRY_EVENT_RE.fullmatch(prefix)
    if parry and rest:
        kind = parry.group("kind")
        return rest, None, InputStep(raw=prefix, requirement=PARRY_SUCCESS + (f"_{kind.upper()}" if kind else ""))
    requirement = _prefix_requirement(prefix, known)
    if requirement is None or not rest or "." in rest:
        return fragment, None, None
    return rest, requirement, None


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
    state_match = re.match(r"^(ws|ss|bt|fc|wr|ch)\+?(?=[udfbnqhc1-4])", token, re.IGNORECASE)
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
    direction_names = {value.lower() for value in _DIRECTIONS}
    alternative_note = None
    for part in parts:
        options = part.replace("*", "").split("/")
        if len(options) > 1 and all(option.lower() in direction_names for option in options):
            # "b/db": either direction works; the first is sent.
            alternative_note = f"alternative directions: {part}"
            part = options[0]
        normalized = part.replace("/", "").replace("*", "")
        direction_candidate = normalized.lower()
        if direction_candidate in direction_names:
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
    warning = f"unparsed input fragment: {raw}" if uncertain else alternative_note
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

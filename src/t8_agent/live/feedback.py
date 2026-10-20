"""Screen-derived execution feedback, side-switch detection, and replay logs.

Everything here uses only what the capture shows: per-side motion, HUD health
changes, and the colors of the two fighters. Where the screen cannot separate
two outcomes (a whiff and a block both leave health unchanged), the result
says so explicitly instead of picking one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from t8_agent.vision.temporal import VisualEstimate


FRAME_SECONDS = 1.0 / 60.0

# Execution outcomes.
FAILED = "failed"                    # the controller refused or could not send it
NOT_ACCEPTED = "not_accepted"        # no own animation started in the startup window
INTERRUPTED = "interrupted"          # we were hit before the move could become active
HIT = "hit"                          # the opponent lost health in the active window
NO_HIT = "whiffed_or_blocked"        # accepted, no damage; whiff and block look alike
PENDING = "pending"


@dataclass
class CommandAttempt:
    candidate_id: str
    sent_at: float
    startup_min: int | None
    startup_max: int | None
    sent: bool
    outcome: str = PENDING
    resolved_at: float | None = None
    accepted_at: float | None = None
    events: list[str] = field(default_factory=list)


class ExecutionFeedback:
    """Classifies each sent command from the frames that follow it.

    Startup frames come from the compiled move (unknown startup uses
    ``default_startup``). ``tolerance_frames`` covers capture and input latency.
    """

    def __init__(self, *, player: int = 1, motion_threshold: float = 0.015,
                 tolerance_frames: int = 8, default_startup: int = 20, active_frames: int = 6) -> None:
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        self.player = player
        self.motion_threshold = motion_threshold
        self.tolerance = tolerance_frames * FRAME_SECONDS
        self.default_startup = default_startup
        self.active = active_frames * FRAME_SECONDS
        self.current: CommandAttempt | None = None
        self.history: list[CommandAttempt] = []

    def start(self, candidate_id: str, *, sent: bool, startup: dict[str, int] | None = None,
              now: float | None = None) -> CommandAttempt:
        if self.current is not None and self.current.outcome == PENDING:
            self._resolve(self.current, NO_HIT if self.current.accepted_at else NOT_ACCEPTED,
                          self.current.sent_at, "superseded")
        attempt = CommandAttempt(candidate_id, time.perf_counter() if now is None else now,
                                 None if startup is None else int(startup["min"]),
                                 None if startup is None else int(startup["max"]), sent)
        self.history.append(attempt)
        if not sent:
            self._resolve(attempt, FAILED, attempt.sent_at, "controller did not send")
            self.current = None
        else:
            self.current = attempt
        return attempt

    def update(self, estimate: VisualEstimate, *, now: float | None = None) -> CommandAttempt | None:
        """Feeds one frame; returns the attempt if it resolved on this frame."""

        attempt = self.current
        if attempt is None:
            return None
        now = time.perf_counter() if now is None else now
        elapsed = now - attempt.sent_at
        own_motion, own_hit, opponent_hit = (
            (estimate.p1_motion, estimate.p1_hit_event, estimate.p2_hit_event) if self.player == 1 else
            (estimate.p2_motion, estimate.p2_hit_event, estimate.p1_hit_event))
        startup_min = (attempt.startup_min or self.default_startup) * FRAME_SECONDS
        startup_max = (attempt.startup_max or self.default_startup) * FRAME_SECONDS
        if attempt.accepted_at is None and own_motion >= self.motion_threshold:
            attempt.accepted_at = now
            attempt.events.append(f"motion@{elapsed:.3f}")
        if own_hit and elapsed <= startup_max + self.tolerance:
            return self._resolve(attempt, INTERRUPTED, now, f"own hit@{elapsed:.3f}")
        if opponent_hit and startup_min - self.tolerance <= elapsed <= startup_max + self.active + self.tolerance:
            return self._resolve(attempt, HIT, now, f"opponent hit@{elapsed:.3f}")
        if attempt.accepted_at is None and elapsed > startup_max + self.tolerance:
            return self._resolve(attempt, NOT_ACCEPTED, now, "no own motion in the startup window")
        if elapsed > startup_max + self.active + self.tolerance:
            return self._resolve(attempt, NO_HIT, now, "no damage in the active window")
        return None

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for attempt in self.history:
            counts[attempt.outcome] = counts.get(attempt.outcome, 0) + 1
        return counts

    def _resolve(self, attempt: CommandAttempt, outcome: str, now: float, reason: str) -> CommandAttempt:
        attempt.outcome = outcome
        attempt.resolved_at = now
        attempt.events.append(reason)
        if self.current is attempt:
            self.current = None
        return attempt


class SideSwitchDetector:
    """Detects that the fighters swapped screen sides from their colors.

    At a round start the colors in each body region are stored as references.
    When the left region keeps looking like the right reference (and vice
    versa) for ``confirm_frames`` frames, the sides switched. The manual F7
    override stays authoritative: call ``reset_reference`` after using it.
    """

    def __init__(self, *, confirm_frames: int = 6, margin: float = 0.7, bins: int = 4) -> None:
        self.confirm_frames = confirm_frames
        self.margin = margin
        self.bins = bins
        self.reset_reference()

    def reset_reference(self) -> None:
        self.reference: tuple[np.ndarray, np.ndarray] | None = None
        self.pending = 0

    def update(self, left_crop: np.ndarray, right_crop: np.ndarray) -> bool:
        left, right = self._signature(left_crop), self._signature(right_crop)
        if left is None or right is None:
            self.pending = 0
            return False
        if self.reference is None:
            self.reference = (left, right)
            return False
        reference_left, reference_right = self.reference
        same = _distance(left, reference_left) + _distance(right, reference_right)
        swapped = _distance(left, reference_right) + _distance(right, reference_left)
        if swapped < self.margin * same:
            self.pending += 1
            if self.pending >= self.confirm_frames:
                self.reference = (left, right)
                self.pending = 0
                return True
        else:
            self.pending = 0
        return False

    def _signature(self, crop: np.ndarray) -> np.ndarray | None:
        crop = np.asarray(crop)
        if crop.ndim != 3 or crop.shape[2] < 3 or crop.size == 0:
            return None
        # A color histogram needs a sample, not every pixel: a half-screen body region is
        # ~800k pixels, which cost ~45 ms per side per decision. A grid of ~40k keeps the
        # signature and costs a few milliseconds.
        stride = max(1, int((crop.shape[0] * crop.shape[1] / 40_000) ** 0.5))
        crop = crop[::stride, ::stride]
        pixels = crop[..., :3].reshape(-1, 3).astype(np.int16)
        bright, dark = pixels.max(axis=1), pixels.min(axis=1)
        foreground = pixels[(bright > 45) & (bright - dark > 18)]
        if len(foreground) < 16:
            return None
        quantized = np.clip(foreground * self.bins // 256, 0, self.bins - 1)
        codes = (quantized[:, 0] * self.bins + quantized[:, 1]) * self.bins + quantized[:, 2]
        histogram = np.bincount(codes, minlength=self.bins ** 3).astype(np.float64)
        return histogram / histogram.sum()


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(first - second).sum())


class ReplayRecorder:
    """Writes one JSON line per decision, controller change, and outcome."""

    def __init__(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.write("session", **(metadata or {}))

    def write(self, kind: str, **values: Any) -> None:
        record = {"kind": kind, "time": values.pop("time", time.perf_counter()), **values}
        self._handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
        self._handle.flush()

    def decision(self, *, estimate: VisualEstimate, candidate: str, sent: bool, legal_count: int,
                 expected_inputs: Iterable[Iterable[str]] | None = None, time_value: float | None = None) -> None:
        self.write("decision", time=time.perf_counter() if time_value is None else time_value,
                   estimate=estimate.to_dict(), candidate=candidate, sent=sent, legal_count=legal_count,
                   expected_inputs=None if expected_inputs is None else [sorted(stage) for stage in expected_inputs])

    def controller(self, time_value: float, held: frozenset[str], action: str | None) -> None:
        self.write("controller", time=time_value, held=sorted(held), action=action)

    def outcome(self, attempt: CommandAttempt) -> None:
        self.write("outcome", time=attempt.resolved_at, **asdict(attempt))

    def close(self) -> None:
        self._handle.close()


def compare_replay(path: str | Path) -> dict[str, Any]:
    """Expected versus observed results of a recorded session.

    For each sent decision it checks that the controller actually pressed the
    expected inputs (in order) before the next decision, and tallies the
    screen-observed outcomes per candidate.
    """

    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = [record for record in records if record["kind"] == "decision"]
    presses = [record for record in records if record["kind"] == "controller"]
    outcomes = [record for record in records if record["kind"] == "outcome"]
    mismatches: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        if not decision["sent"] or not decision.get("expected_inputs"):
            continue
        end = decisions[index + 1]["time"] if index + 1 < len(decisions) else float("inf")
        observed = [record["held"] for record in presses
                    if decision["time"] <= record["time"] < end and record["held"]]
        expected = [stage for stage in decision["expected_inputs"] if stage]
        if not _is_subsequence(expected, observed):
            mismatches.append({"time": decision["time"], "candidate": decision["candidate"],
                               "expected": expected, "observed": observed})
    per_candidate: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        counts = per_candidate.setdefault(outcome["candidate_id"], {})
        counts[outcome["outcome"]] = counts.get(outcome["outcome"], 0) + 1
    totals: dict[str, int] = {}
    for counts in per_candidate.values():
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
    sent = sum(1 for decision in decisions if decision["sent"])
    return {"decisions": len(decisions), "sent": sent, "input_mismatches": mismatches,
            "outcomes": totals, "per_candidate": per_candidate}


def _is_subsequence(expected: list[list[str]], observed: list[list[str]]) -> bool:
    position = 0
    for stage in observed:
        if position < len(expected) and sorted(stage) == sorted(expected[position]):
            position += 1
    return position == len(expected)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")

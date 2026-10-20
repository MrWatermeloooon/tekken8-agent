"""Provenance for every manually measured or reviewed value.

A measured field, a source correction, and a Practice validation record each
carry the same provenance block, so any value can be traced to the game patch
it was taken on, when, how, how confident the measurer was, and whether a
second person reviewed it:

    patch: "2.01.01"            # in-game version string, or "unknown"
    recorded_at: "2026-09-23"   # ISO date the value was measured or retrieved
    method: practice_mode       # one of METHODS
    confidence: high            # one of CONFIDENCE_LEVELS
    reviewer: "name or 'automated review'"
    review_status: unreviewed   # one of REVIEW_STATUSES

Measurement files (data/measurements/<slug>.yaml) may set a file-level
``provenance`` block and a per-move ``provenance`` block; a field-level block
overrides both. A field whose merged provenance is incomplete is rejected
rather than silently trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml


PROVENANCE_FIELDS = ("patch", "recorded_at", "method", "confidence", "reviewer", "review_status")
METHODS = frozenset({
    "practice_mode",         # measured in offline Practice mode (frame display, distance readout)
    "input_history",         # confirmed from Practice-mode input history
    "video_frame_count",     # counted frame-by-frame from a capture
    "source_document",       # taken from a cited frame-data source or guide
    "derived",               # computed from other recorded values
})
CONFIDENCE_LEVELS = ("low", "medium", "high")
REVIEW_STATUSES = ("unreviewed", "self_checked", "peer_reviewed")
# Fields the character gate requires for every move (the catalog's "measurements").
MEASURED_FIELDS = ("active_frames", "range", "tracking_left", "tracking_right", "pushback", "collision")
# Every field a measurement file may set: the gate fields plus the values the
# full-combat binder (tools/export_full_combat_bindings.py) reads when a move's
# notes leave them unstated.
ACCEPTED_FIELDS = MEASURED_FIELDS + (
    "travel", "hit_frames", "chip_block", "chip_block_heat", "self_damage", "self_recoverable",
    "restore_health_hit", "restore_recoverable_hit", "restore_recoverable_block", "throw_break")


@dataclass(frozen=True)
class MeasurementSet:
    values: dict[str, dict[str, Any]]           # stable_id -> field -> value
    provenance: dict[str, dict[str, dict]]      # stable_id -> field -> provenance block
    issues: tuple[str, ...]                     # rejected fields and why
    stale: frozenset[str]                       # stable_ids invalidated by a patch update


def provenance_issues(block: Mapping[str, Any] | None) -> list[str]:
    """Returns why a provenance block is incomplete or invalid (empty when valid)."""

    block = block or {}
    issues = [f"missing {name}" for name in PROVENANCE_FIELDS if block.get(name) in (None, "")]
    if block.get("recorded_at") not in (None, ""):
        try:
            date.fromisoformat(str(block["recorded_at"]))
        except ValueError:
            issues.append(f"recorded_at is not an ISO date: {block['recorded_at']}")
    if block.get("method") not in (None, "") and str(block["method"]) not in METHODS:
        issues.append(f"unknown method: {block['method']}")
    if block.get("confidence") not in (None, "") and str(block["confidence"]) not in CONFIDENCE_LEVELS:
        issues.append(f"unknown confidence: {block['confidence']}")
    if block.get("review_status") not in (None, "") and str(block["review_status"]) not in REVIEW_STATUSES:
        issues.append(f"unknown review_status: {block['review_status']}")
    return issues


def require_provenance(block: Mapping[str, Any] | None, context: str) -> dict[str, Any]:
    issues = provenance_issues(block)
    if issues:
        raise ValueError(f"{context}: incomplete provenance ({'; '.join(issues)})")
    return {name: _plain(block[name]) for name in PROVENANCE_FIELDS}


def load_measurements(path: str | Path) -> MeasurementSet:
    """Loads a measurement file, keeping only fields with complete provenance.

    Moves marked ``stale`` by tools/patch_update.py count as unmeasured.
    """

    path = Path(path)
    if not path.exists():
        return MeasurementSet({}, {}, (), frozenset())
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    file_block = dict(document.get("provenance") or {})
    values: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, dict]] = {}
    issues: list[str] = []
    stale: set[str] = set()
    for stable_id, entry in (document.get("moves") or {}).items():
        stable_id = str(stable_id)
        entry = dict(entry or {})
        if entry.get("stale"):
            stale.add(stable_id)
            continue
        move_block = {**file_block, **dict(entry.pop("provenance", None) or {})}
        for field, raw in entry.items():
            if field not in ACCEPTED_FIELDS:
                issues.append(f"{stable_id}.{field}: unknown measured field")
                continue
            if isinstance(raw, Mapping):
                value = raw.get("value")
                block = {**move_block, **{key: raw[key] for key in PROVENANCE_FIELDS if key in raw}}
            else:
                value, block = raw, move_block
            if value is None:
                issues.append(f"{stable_id}.{field}: missing value")
                continue
            problems = provenance_issues(block)
            if problems:
                issues.append(f"{stable_id}.{field}: {'; '.join(problems)}")
                continue
            values.setdefault(stable_id, {})[field] = _plain(value)
            provenance.setdefault(stable_id, {})[field] = {name: _plain(block[name]) for name in PROVENANCE_FIELDS}
    return MeasurementSet(values, provenance, tuple(issues), frozenset(stale))


def _plain(value: Any) -> Any:
    """YAML dates become ISO strings so compiled catalogs stay JSON-canonical."""

    if isinstance(value, date):
        return value.isoformat()
    return value

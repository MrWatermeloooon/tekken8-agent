"""Schema-migration and catalog-hash compatibility checks.

Every persisted format names its schema version, and each loader must accept
exactly the versions it can read and reject the rest with a clear error. The
catalog identity fixture pins the hash that existing full-roster checkpoints
are bound to, so a catalog change (a game patch, a parser fix, a correction)
is a deliberate, recorded event rather than a silent resume break.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import yaml

from t8_agent.live.full_move import read_trainer_identity
from t8_agent.moves.catalog import CATALOG_SCHEMA_VERSION, compile_catalog, load_compiled_catalog
from t8_agent.moves.identity import IDENTITY_SCHEMA_VERSION, load_identity
from t8_agent.moves.provenance import load_measurements
from t8_agent.moves.validation import load_validation_ledger


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "contracts" / "catalog_identity.json"


def test_catalog_identity_matches_the_recorded_fixture():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    committed = load_compiled_catalog(REPO_ROOT / "data" / "generated" / "full_move_catalog.json")
    rebuilt = compile_catalog(REPO_ROOT / "data")
    assert rebuilt.catalog_sha256 == committed.catalog_sha256, \
        "data/generated is stale: rerun tools/compile_full_move_catalog.py"
    assert committed.catalog_sha256 == fixture["catalog_sha256"], (
        "the move catalog hash changed; full-roster checkpoints trained on the old catalog can no longer "
        "resume. If intended, update tests/fixtures/contracts/catalog_identity.json and add a history entry.")
    assert fixture["history"][-1]["catalog_sha256"] == fixture["catalog_sha256"]
    assert len({entry["catalog_sha256"] for entry in fixture["history"]}) == len(fixture["history"])
    assert (committed.document["schema_version"], committed.document["roster_as_of"],
            committed.document["move_count"]) == (fixture["schema_version"], fixture["roster_as_of"],
                                                  fixture["move_count"])


def test_native_catalog_csv_carries_the_same_identity():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with (REPO_ROOT / "data" / "generated" / "full_move_catalog.csv").open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        first = dict(zip(header, handle.readline().strip().split(",")))
    assert first["catalog_sha256"] == fixture["catalog_sha256"]
    assert int(first["schema_version"]) == fixture["schema_version"]


def test_catalog_loader_rejects_other_schema_versions_and_tampering(tmp_path):
    document = json.loads((REPO_ROOT / "data" / "generated" / "full_move_catalog.json").read_text(encoding="utf-8"))
    for version in (CATALOG_SCHEMA_VERSION - 1, CATALOG_SCHEMA_VERSION + 1):
        path = tmp_path / f"v{version}.json"
        path.write_text(json.dumps({**document, "schema_version": version}), encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported full move catalog schema"):
            load_compiled_catalog(path)
    tampered = dict(document)
    tampered["moves"] = [dict(document["moves"][0], name="tampered")] + document["moves"][1:]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_compiled_catalog(path)


def test_validation_ledger_versions(tmp_path):
    assert load_validation_ledger(tmp_path / "missing.json")["schema_version"] == 1
    path = tmp_path / "jun.json"
    path.write_text(json.dumps({"schema_version": 1, "moves": {"jun:1": {"status": "pass"}}}), encoding="utf-8")
    assert load_validation_ledger(path)["moves"]["jun:1"]["status"] == "pass"
    path.write_text(json.dumps({"schema_version": 2, "moves": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported validation ledger schema"):
        load_validation_ledger(path)


def test_identity_registry_versions(tmp_path):
    (tmp_path / "identity").mkdir()
    current = json.loads((REPO_ROOT / "data" / "identity" / "jun.json").read_text(encoding="utf-8"))
    assert current["schema_version"] == IDENTITY_SCHEMA_VERSION
    (tmp_path / "identity" / "jun.json").write_text(
        json.dumps({**current, "schema_version": IDENTITY_SCHEMA_VERSION + 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_identity(tmp_path, "jun")


def test_measurement_files_accept_both_value_forms(tmp_path):
    provenance = {"patch": "2.01.01", "recorded_at": "2026-09-30", "method": "practice_mode",
                  "confidence": "high", "reviewer": "r", "review_status": "unreviewed"}
    path = tmp_path / "jun.yaml"
    # Plain values under a file-level block, and per-field mappings, are both current forms.
    path.write_text(yaml.safe_dump({"provenance": provenance, "moves": {
        "jun:1": {"range": 1.1}, "jun:2": {"range": {"value": 0.8, "confidence": "low"}}}}), encoding="utf-8")
    assert load_measurements(path).values == {"jun:1": {"range": 1.1}, "jun:2": {"range": 0.8}}
    # Files written before provenance existed load, but every field is rejected.
    path.write_text(yaml.safe_dump({"moves": {"jun:1": {"range": 1.1}}}), encoding="utf-8")
    legacy = load_measurements(path)
    assert legacy.values == {} and legacy.issues


@pytest.mark.parametrize("format_version", [6, 7, 8])
def test_trainer_state_formats_that_resume_expose_their_identity(tmp_path, format_version):
    header = struct.pack("<8sIQQdQQQiQ4fQ8f4Q3Bi", b"T8RUNV2\0", format_version, 7, 1, 0.5, 1, 16, 8, 1, 16,
                         *([0.1] * 4), 1, *([0.1] * 8), 1, 1, 1, 1, 0, 1, 1, 0)
    strings = b"".join(struct.pack("<Q", len(value)) + value for value in (b"f" * 64, b"2026-07-21", b"jun"))
    path = tmp_path / "state.t8state"
    path.write_bytes(header + strings)
    identity = read_trainer_identity(path)
    assert (identity.format_version, identity.learner_character) == (format_version, "jun")
    path.write_bytes(header.replace(struct.pack("<I", format_version), struct.pack("<I", 5), 1) + strings)
    with pytest.raises(ValueError, match="format 6-8"):
        read_trainer_identity(path)


def test_recorded_runs_are_flagged_against_the_current_catalog():
    """Trainer states from before a catalog change must be recognisably incompatible."""

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    known = {entry["catalog_sha256"] for entry in fixture["history"]}
    state = REPO_ROOT / "runs" / "stability_ab_2026-09-24" / "checkpoints" / "update_100.t8state"
    if not state.exists():
        pytest.skip("local training run not present")
    identity = read_trainer_identity(state)
    assert identity.catalog_sha256 in known
    assert identity.catalog_sha256 != fixture["catalog_sha256"]

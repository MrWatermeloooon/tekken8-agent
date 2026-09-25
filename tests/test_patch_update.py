from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from t8_agent.moves.catalog import _load_corrections
from t8_agent.moves.identity import bootstrap_identity, source_keys, stable_numbers, update_identity


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move(source_id: str, index: int, command: str, **fields) -> dict:
    row = {"id": command, "source_id": source_id, "source_index": index, "command": command, "name": "",
           "hit_level": "m", "damage": "10", "startup": "i12", "recovery": "r20", "block": "-2", "hit": "+5",
           "counter_hit": "", "tier": "specialist", "tags": [], "notes": ""}
    row.update(fields)
    return row


def _snapshot(moves: list[dict], retrieved: str = "2026-07-21") -> dict:
    return {"character_id": "tst", "display_name": "Test", "game": "tekken_8", "source_url": "",
            "retrieved_at": retrieved, "source_sha256": retrieved.replace("-", "") * 3, "data_status": "available",
            "stances": [], "moves": moves}


def test_identity_survives_insertion_removal_and_return():
    old = [_move("T-1", 1, "1"), _move("T-2", 2, "2"), _move("T-3", 3, "3")]
    registry = bootstrap_identity("tst", old)
    assert stable_numbers("tst", old, registry) == [1, 2, 3]
    # The source inserts a move before T-2 and drops T-3: its move numbers shift, stable IDs do not.
    new = [_move("T-1", 1, "1"), _move("T-new", 2, "1+2"), _move("T-2", 3, "2")]
    updated, added, retired = update_identity(registry, new)
    assert added == ["T-new"] and retired == ["T-3"]
    assert stable_numbers("tst", new, updated) == [1, 4, 2]
    # A retired move that returns keeps its number; numbers are never reused.
    back, added, _ = update_identity(updated, new + [_move("T-3", 4, "3")])
    assert back["moves"]["T-3"] == 3 and back["next_number"] == 5
    with pytest.raises(ValueError, match="without a stable ID"):
        stable_numbers("tst", new + [_move("T-unknown", 9, "9")], updated)


def test_repeated_source_ids_get_occurrence_keys():
    rows = [_move("T-3+4", 1, "3+4"), _move("T-3+4", 2, "3+4"), _move("T-1", 3, "1")]
    assert source_keys(rows) == ["T-3+4", "T-3+4#2", "T-1"]


def test_repository_catalog_ids_come_from_the_registries():
    catalog = json.loads((REPO_ROOT / "data" / "generated" / "full_move_catalog.json").read_text(encoding="utf-8"))
    jun = [move for move in catalog["moves"] if move["stable_id"].startswith("jun:")]
    registry = json.loads((REPO_ROOT / "data" / "identity" / "jun.json").read_text(encoding="utf-8"))
    assert {move["stable_id"] for move in jun} == {f"jun:{number}" for number in registry["moves"].values()}


def test_diff_reports_real_changes_and_ignores_whitespace():
    patch = _tool("patch_update")
    old = _snapshot([_move("T-1", 1, "1", notes="* Homing\n* Low crush 14~33"), _move("T-2", 2, "2"),
                     _move("T-3", 3, "3", notes="* Spike")])
    new = _snapshot([_move("T-1", 1, "1", notes="* Homing\n* Low crush 14~46", recovery="r18"),
                     _move("T-3", 2, "3", notes="*   Spike  "), _move("T-4", 3, "4")], "2026-09-23")
    diff = patch.diff_snapshots(old, new)
    assert diff["added"] == ["T-4"] and diff["removed"] == ["T-2"]
    assert set(diff["changed"]) == {"T-1"}
    assert diff["changed"]["T-1"]["recovery"] == {"old": "r20", "new": "r18"}
    assert diff["changed"]["T-1"]["notes"] == {"removed": ["Low crush 14~33"], "added": ["Low crush 14~46"]}
    assert diff["source_renumbered"] == ["T-3"]


def test_apply_patch_invalidates_everything_that_depended_on_changed_moves(tmp_path):
    patch = _tool("patch_update")
    data = tmp_path / "data"
    (data / "characters").mkdir(parents=True)
    old = _snapshot([_move("T-1", 1, "1"), _move("T-2", 2, "2", startup="i16~15"), _move("T-3", 3, "3"),
                     _move("T-4", 4, "4")])
    (data / "characters" / "tst.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
    (data / "validation").mkdir()
    (data / "validation" / "tst.json").write_text(json.dumps({
        "schema_version": 1,
        "moves": {"tst:1": {"status": "pass"}, "tst:2": {"status": "pass"}, "tst:3": {"status": "pass"},
                  "tst:4": {"status": "pass"}},
        "scenarios": {"hit": "pass", "block": "pending"},
        "routes": {"midscreen": "pass", "wall": "pass"}}), encoding="utf-8")
    (data / "measurements").mkdir()
    (data / "measurements" / "tst.yaml").write_text(yaml.safe_dump(
        {"moves": {"tst:1": {"range": 1.2}, "tst:4": {"range": 0.9}}}), encoding="utf-8")
    (data / "corrections").mkdir()
    (data / "corrections" / "tst.yaml").write_text("# header kept\n" + yaml.safe_dump({"corrections": [
        {"source_id": "T-2", "fields": {"startup": "i15~16"}, "source_fields": {"startup": "i16~15"},
         "reason": "r", "evidence": ["e"], "reviewer": "x"},
        {"source_id": "T-4", "fields": {"block": "-6"}, "source_fields": {"block": "-2"},
         "reason": "r", "evidence": ["e"], "reviewer": "x"}]}), encoding="utf-8")
    (data / "character_modules" / "tst").mkdir(parents=True)
    (data / "character_modules" / "tst" / "routes.yaml").write_text(yaml.safe_dump({"routes": [
        {"name": "uses_one", "category": "midscreen", "steps": ["1", "4"]},
        {"name": "uses_three", "category": "wall", "steps": ["3"]}]}), encoding="utf-8")

    # T-1 changes, T-2's source now matches its correction, T-3 is only renamed, T-4's corrected field
    # changes to something else, and T-5 is new.
    new = _snapshot([_move("T-1", 1, "1", recovery="r18"), _move("T-2", 2, "2", startup="i15~16"),
                     _move("T-3", 3, "3", name="Renamed"), _move("T-4", 4, "4", block="-4"),
                     _move("T-5", 5, "5")], "2026-09-23")
    record = patch.apply_patch("tst", new, data, today="2026-09-23")

    ledger = json.loads((data / "validation" / "tst.json").read_text(encoding="utf-8"))
    assert ledger["moves"]["tst:1"]["status"] == "stale" and ledger["moves"]["tst:1"]["history"] == [{"status": "pass"}]
    assert ledger["moves"]["tst:3"] == {"status": "pass"}, "a rename does not invalidate"
    assert ledger["scenarios"]["hit"]["status"] == "stale" and ledger["scenarios"]["block"] == "pending"
    assert ledger["routes"]["midscreen"]["status"] == "stale" and ledger["routes"]["wall"] == "pass"
    measurements = yaml.safe_load((data / "measurements" / "tst.yaml").read_text(encoding="utf-8"))
    assert measurements["moves"]["tst:1"]["stale"] and measurements["moves"]["tst:4"]["stale"]
    exporter = _tool("export_full_combat_bindings")
    assert set(exporter.load_measurements(data / "measurements" / "tst.yaml")) == set(), "stale measurements count as missing"
    corrections_text = (data / "corrections" / "tst.yaml").read_text(encoding="utf-8")
    corrections = {entry["source_id"]: entry for entry in yaml.safe_load(corrections_text)["corrections"]}
    assert corrections_text.startswith("# header kept")
    assert corrections["T-2"]["status"] == "resolved_upstream"
    assert corrections["T-4"]["status"] == "needs_review"
    identity = json.loads((data / "identity" / "tst.json").read_text(encoding="utf-8"))
    assert identity["moves"]["T-5"] == 5
    assert (data / record["archived_snapshot"]).exists()
    assert (data / "patches" / "tst" / "2026-09-23.json").exists()


def test_corrections_follow_patch_statuses():
    rows = [_move("T-1", 1, "1", startup="i16~15")]
    base = {"source_id": "T-1", "fields": {"startup": "i15~16"}, "reason": "r", "evidence": ["e"], "reviewer": "x"}

    def load(tmp_entry, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"corrections": [tmp_entry]}), encoding="utf-8")
        return _load_corrections(path, rows)

    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert "T-1" in load({**base, "source_fields": {"startup": "i16~15"}}, root)
        assert load({**base, "status": "resolved_upstream"}, root) == {}
        with pytest.raises(ValueError, match="needs review"):
            load({**base, "status": "needs_review"}, root)
        with pytest.raises(ValueError, match="written against"):
            load({**base, "source_fields": {"startup": "i14"}}, root)


def test_importer_normalizes_the_newer_api_format():
    importer = _tool("import_roster_frame_data")
    row = {"command": "1,1", "hitLevel": "h, m", "recovery": "20", "recoveryState": "IZU",
           "tags": {"intr": "7"}, "transitions": ["IZU"], "wavuId": "Jun-1,1", "moveNumber": 7}
    converted = importer._convert_move(row)
    assert converted["recovery"] == "r20 IZU"
    assert "intr" in converted["tags"] and converted["transitions"] == ["IZU"]
    assert importer._convert_move({**row, "recovery": "r28 IZU", "tags": ["he"]})["recovery"] == "r28 IZU"
    assert importer._convert_move({**row, "recovery": "", "recoveryState": "IZU"})["recovery"] == "IZU"

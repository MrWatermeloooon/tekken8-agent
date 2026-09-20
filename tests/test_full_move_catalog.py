from __future__ import annotations

import json
from pathlib import Path

import pytest

from t8_agent.moves.catalog import ACTION_FEATURE_SIZE, compile_catalog, load_compiled_catalog, write_catalog
from t8_agent.moves.notation import parse_command
from t8_agent.moves.validation import evaluate_character_gate
from t8_agent.moves.legal import MoveRuntimeState, UNIVERSAL_ACTIONS, legal_action_mask


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command,requirements,steps", [
    ("1", (), 1),
    ("df+1", (), 1),
    ("1,1+2", (), 2),
    ("H.GEN.1,1+2", ("H", "GEN"), 2),
    ("FC.df+1", ("FC",), 1),
    ("f,F+2", (), 2),
    ("ws1+2", (), 1),
    ("ss2", (), 1),
    ("3~4", (), 2),
])
def test_command_parser_core_notation(command, requirements, steps):
    parsed = parse_command(command, {"GEN"})
    assert parsed.parser_status == "parsed"
    assert parsed.requirements == requirements
    assert len(parsed.steps) == steps


def test_command_parser_keeps_per_step_state_requirements():
    assert parse_command("ws1").steps[0].requirement == "WS"
    assert parse_command("ss2").steps[0].requirement == "SS"


def test_just_frame_separator_marks_the_following_input():
    parsed = parse_command("3:4")
    assert [step.just_frame for step in parsed.steps] == [False, True]


def test_command_parser_preserves_unknown_fragments_for_review():
    parsed = parse_command("GEN.???", {"GEN"})
    assert parsed.parser_status == "needs_review"
    assert parsed.steps[0].uncertain
    assert parsed.warnings


def test_full_catalog_contains_every_imported_source_row(tmp_path):
    catalog = compile_catalog(REPO_ROOT / "data")
    assert len(catalog.characters) == 42
    assert len(catalog.moves) == 6393
    assert sum(character["move_count"] for character in catalog.characters) == 6393
    assert len({move["stable_id"] for move in catalog.moves}) == 6393
    slugs = {character["id"]: character["slug"] for character in catalog.characters}
    assert all(
        move["stable_id"] == f'{slugs[move["character_id"]]}:{move["source_index"]}'
        for move in catalog.moves
    )
    assert all(len(move["action_features"]) == ACTION_FEATURE_SIZE for move in catalog.moves)
    jun = next(character for character in catalog.characters if character["slug"] == "jun")
    bob = next(character for character in catalog.characters if character["slug"] == "bob")
    assert jun["move_count"] == 149
    assert bob["move_count"] == 0 and bob["practice_validation"] == "blocked_missing_source"
    assert not any(character["training_enabled"] for character in catalog.characters)

    output = tmp_path / "catalog.json"
    write_catalog(catalog, output)
    assert load_compiled_catalog(output).catalog_sha256 == catalog.catalog_sha256
    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["moves"][0]["name"] = "corrupted"
    output.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_compiled_catalog(output)


def test_every_move_has_provenance_and_explicit_measurement_gaps():
    catalog = compile_catalog(REPO_ROOT / "data")
    for move in catalog.moves:
        assert move["provenance"]["source_url"]
        assert move["provenance"]["retrieved_at"]
        assert move["command"]["raw"]
        assert set(move["measurements"]) == {
            "active_frames", "range", "tracking_left", "tracking_right", "pushback", "collision"
        }
        assert move["validation"]["practice"] == "pending"


def test_impossible_source_frames_are_preserved_but_blocked():
    catalog = compile_catalog(REPO_ROOT / "data")
    inconsistent = [move for move in catalog.moves
                    if move["validation"]["source_consistency"] == "blocked"]
    assert inconsistent
    assert any("startup_range_descends" in move["validation"]["issues"]
               for move in inconsistent)
    assert all(not move["validation"]["issues"] == [] for move in inconsistent)


def test_every_move_has_an_explicit_legal_state_rule():
    catalog = compile_catalog(REPO_ROOT / "data")
    assert all(move["legal_state"]["source"] == "command_notation" for move in catalog.moves)
    assert all(move["legal_state"]["posture"] for move in catalog.moves)


def test_character_gate_never_enables_unvalidated_data(tmp_path):
    catalog = compile_catalog(REPO_ROOT / "data")
    jun = evaluate_character_gate(catalog, "jun", tmp_path)
    bob = evaluate_character_gate(catalog, "bob", tmp_path)
    assert not jun.ready and jun.move_count == 149
    assert jun.parser_pending > 0 and jun.measurement_pending == 149
    assert not bob.ready and "missing documented move source" in bob.blockers


def test_character_conditioned_mask_has_variable_candidates_and_busy_fallback():
    catalog = compile_catalog(REPO_ROOT / "data")
    jun = legal_action_mask(catalog, "jun", MoveRuntimeState(), require_validated=False)
    yoshimitsu = legal_action_mask(
        catalog, "yoshimitsu", MoveRuntimeState(), require_validated=False)
    assert jun.shape == (len(UNIVERSAL_ACTIONS) + 149,)
    assert yoshimitsu.shape == (len(UNIVERSAL_ACTIONS) + 312,)
    assert jun[:len(UNIVERSAL_ACTIONS)].all()
    busy = legal_action_mask(
        catalog, "jun", MoveRuntimeState(busy=True), require_validated=False)
    assert busy[0] and busy.sum() == 1


def test_strict_mask_rejects_every_unvalidated_move():
    catalog = compile_catalog(REPO_ROOT / "data")
    mask = legal_action_mask(catalog, "jun", MoveRuntimeState(), require_validated=True)
    assert mask[:len(UNIVERSAL_ACTIONS)].all()
    assert not mask[len(UNIVERSAL_ACTIONS):].any()

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
    # Notation and source consistency are resolved (data/corrections/jun.yaml);
    # measurement and Practice validation still block training.
    assert jun.parser_pending == 0 and jun.source_conflicts == 0
    assert jun.measurement_pending == 149 and jun.practice_pending == 149
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


def test_notation_parses_positional_throws_wall_prefix_and_parry_outcomes():
    from t8_agent.moves.notation import parse_command

    back = parse_command("Back throw")
    assert back.parser_status == "parsed" and back.requirements == ("OPPONENT_BACK_TURNED",)
    assert [step.buttons for step in back.steps] == [(1, 3)]
    assert parse_command("Right Throw").requirements == ("OPPONENT_RIGHT_SIDE",)
    wall = parse_command("(Back to wall).b,b,UB")
    assert wall.parser_status == "parsed" and wall.requirements == ("BACK_TO_WALL",)
    assert [(step.direction, step.hold) for step in wall.steps] == [("b", False), ("b", False), ("UB", True)]
    parry = parse_command("b+1+3,P")
    assert parry.parser_status == "parsed" and parry.steps[0].buttons == (1, 3)
    assert parry.steps[1].requirement == "PARRY_SUCCESS" and parry.steps[1].buttons == ()
    stance = parse_command("GEN.P (Low)", {"GEN"})
    assert stance.requirements == ("GEN",) and stance.steps[0].requirement == "PARRY_SUCCESS_LOW"
    # Unknown parenthesized situations and unhandled continuations stay unresolved.
    assert parse_command("(During Enemy wall stun) 1+3").parser_status != "parsed"
    assert parse_command("b+1+3,P.2").parser_status != "parsed"


def test_corrections_are_applied_with_provenance():
    catalog = compile_catalog(REPO_ROOT / "data")
    izumo_3 = next(move for move in catalog.moves if move["stable_id"] == "jun:127")
    assert izumo_3["startup"] == {"min": 15, "max": 16}
    assert izumo_3["validation"]["source_consistency"] == "valid"
    assert izumo_3["validation"]["source"] == "imported+corrected"
    assert izumo_3["corrections"]["evidence"] and izumo_3["corrections"]["status"] == "pending_practice"
    back_throw = next(move for move in catalog.moves if move["stable_id"] == "jun:140")
    assert back_throw["legal_state"]["situations"] == ["OPPONENT_BACK_TURNED"]
    assert back_throw["legal_state"]["stances"] == []


def test_corrections_reject_unreviewed_or_unknown_entries(tmp_path):
    import pytest
    from t8_agent.moves.catalog import _load_corrections

    path = tmp_path / "corrections.yaml"
    rows = [{"source_id": "X-1"}]

    def write(entry: str) -> None:
        path.write_text("corrections:" + chr(10) + entry, encoding="utf-8")

    write("- {source_id: X-1, fields: {startup: i10}}")
    with pytest.raises(ValueError, match="reason, evidence, and reviewer"):
        _load_corrections(path, rows)
    write("- {source_id: X-2, fields: {startup: i10}, reason: r, evidence: [u], reviewer: me}")
    with pytest.raises(ValueError, match="unknown source_id"):
        _load_corrections(path, rows)
    write("- {source_id: X-1, fields: {name: n}, reason: r, evidence: [u], reviewer: me}")
    with pytest.raises(ValueError, match="must replace only"):
        _load_corrections(path, rows)


def test_full_combat_binding_exporter_parses_source_text():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_bindings", REPO_ROOT / "tools" / "export_full_combat_bindings.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    assert exporter.parse_advantage("+12a (+3)") == (12, "launch", False)
    assert exporter.parse_advantage("+0d") == (0, "knockdown", False)
    assert exporter.parse_advantage("+21a~+64a (-5~+38)") == (21, "launch", True)
    assert exporter.parse_advantage("-6") == (-6, "stun", False)
    assert exporter.parse_advantage("") == (None, "stun", False)
    windows = exporter.parse_windows("* Floating state 5~13\n* Low crush 14~33\n* Floating state 34~36")
    assert windows == {"airborne": "5~13|34~36", "low_crush": "14~33"}
    assert exporter.parse_windows("* Parry state 5~")["parry"] == f"5~{exporter.OPEN_WINDOW_END}"
    assert exporter.automatic_transitions("* Transitions to IZU on hit or block") == "Transitions to IZU on hit or block"
    assert exporter.automatic_transitions("* Enter GEN +0 +11g r18 with F\n* Enter MIA wiith B") == ""
    assert exporter.one_line('<div>\n\n* A\n* B\n</div>') == "<div> * A * B </div>"

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
    # Stable IDs come from the identity registries, not the source's move numbers (which can shift).
    slugs = {character["id"]: character["slug"] for character in catalog.characters}
    registries = {slug: json.loads((REPO_ROOT / "data" / "identity" / f"{slug}.json").read_text(encoding="utf-8"))
                  for slug in slugs.values()}
    assert all(
        int(move["stable_id"].split(":")[1]) in registries[slugs[move["character_id"]]]["moves"].values()
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


def _exporter():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_bindings", REPO_ROOT / "tools" / "export_full_combat_bindings.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    return exporter


def test_full_combat_binding_exporter_parses_resource_and_throw_notes():
    exporter = _exporter()
    lines = exporter.note_lines

    assert exporter.parse_heat_dash(lines("* Heat Engager\n* Heat Dash +5, +43d (+35)")) == {
        "heat_dash_block": 5, "heat_dash_hit": 43, "heat_dash_effect": "knockdown"}
    assert exporter.parse_heat_dash(lines("* Heat Dash +67a (+50), +5")) == {
        "heat_dash_block": 5, "heat_dash_hit": 67, "heat_dash_effect": "launch"}
    assert exporter.parse_heat_dash(lines("* Heat Dash +39a (+23a) on hit, +5 on block"))["heat_dash_block"] == 5
    assert exporter.parse_heat_dash(lines("* Heat Dash +42a (+27)"))["heat_dash_block"] == ""

    assert exporter.parse_chip(lines("* 10 chip damage on block")) == {"chip_block": "10", "chip_block_heat": ""}
    assert exporter.parse_chip(lines("* Chip damage (2,7) on block"))["chip_block"] == "2|7"
    assert exporter.parse_chip(lines("* Chip damage on block"))["chip_block"] == exporter.UNKNOWN
    assert exporter.parse_chip(lines("* 6 chip damage on block in heat"))["chip_block_heat"] == "6"
    assert exporter.parse_chip(lines("* Deals 8 (DA:11) chip damage on block"))["chip_block"] == "8"
    # Conditional or on-hit chip is not block chip.
    assert exporter.parse_chip(lines(
        "* -8 frame advantage and 4 chip damage on block after absorbing an attack in power crush state\n"
        "* 12 chip damage on hit\n* Divine Aura: 12 chip damage on block"))["chip_block"] == ""

    recoverable = exporter.parse_recoverable(lines(
        "* Deals 12 damage to self (8 recoverable)\n* Restores 32 recoverable on hit (16 on block)\n"
        "* Removes Recoverable Health"))
    assert (recoverable["self_damage"], recoverable["self_recoverable"]) == (12, 8)
    assert (recoverable["restore_recoverable_hit"], recoverable["restore_recoverable_block"]) == (32, 16)
    assert recoverable["removes_recoverable"] == 1
    heat_only = exporter.parse_recoverable(lines("* Deal 10 recoverable damage to self without Heat"))
    assert (heat_only["self_damage"], heat_only["self_recoverable"], heat_only["self_damage_without_heat"]) == (10, 10, 1)
    restores = exporter.parse_recoverable(lines("* Restores 2 health and 2 recoverable on hit\n* Only deals recoverable damage"))
    assert (restores["restore_health_hit"], restores["restore_recoverable_hit"], restores["recoverable_only"]) == (2, 2, 1)
    assert exporter.parse_recoverable(lines("* Restores recoverable health on hit"))["restore_recoverable_hit"] == "?"

    assert exporter.parse_throw_break(lines("* Homing\n* Throw break 1 or 2")) == "1|2"
    assert exporter.parse_throw_break(lines("* Throw break: 1+2")) == "1+2"
    assert exporter.parse_throw_break(lines("* 2 throw break.")) == "2"
    assert exporter.parse_throw_break(lines("* Unbreakable\n* Side switch")) == "none"
    assert exporter.parse_throw_break(lines("* 1 or 2 throw break, depending on King's input.")) == "?"
    assert exporter.parse_throw_break(lines("* Homing")) == ""

    flags = exporter.parse_flags(lines("* Side switch on break\n* Spike\n* Unparryable\n* Reversal Break"))
    assert flags == {"side_switch_on_hit": 0, "side_switch_on_break": 1, "spike": 1, "unparryable": 1,
                     "reversal_break": 1}
    assert exporter.parse_flags(lines("* Side switch"))["side_switch_on_hit"] == 1
    assert exporter.parse_flags(lines("* Can side switch on hit\n* Spike (CH)"))["side_switch_on_hit"] == 0



def test_full_combat_binding_exporter_parses_character_state():
    exporter = _exporter()
    lines = exporter.note_lines
    stances = {"GEN", "IZU", "MIA"}

    assert exporter.result_state("r28 IZU", stances) == ("IZU", 0)
    assert exporter.result_state("r25 FC", stances) == ("", 1)
    assert exporter.result_state("r31", stances) == ("", 0)
    assert exporter.result_state("r20 XYZ", stances) == ("", 0)

    transitions, used = exporter.stance_transitions(lines(
        "* Transition to r20 MIA on hit only\n* Transition to attack throw on hit"), stances)
    assert transitions == {"result_stance_on_hit": "MIA", "result_stance_on_block": ""}
    assert used == ["Transition to r20 MIA on hit only"]
    both, _ = exporter.stance_transitions(lines("* Transitions to IZU on hit or block"), stances)
    assert both == {"result_stance_on_hit": "IZU", "result_stance_on_block": "IZU"}

    variants = exporter.stance_variants(lines(
        "* Enter GEN +0 +11g r18 with F\n* Enter MIA -6, +5 r18 wiith B\n* Enter SS r16 with u_d\n"
        "* Transition to +9, +26a (+16) GEN with F\n* Transition to r24 FC with D\n"
        "* Transition to r22 FC with D on whiff or block"), stances)
    assert [(v["target"], v["block"], v["hit"], v["recovery"], v["input"]) for v in variants] == [
        ("GEN", "+0", "+11g", 18, "F"), ("MIA", "-6", "+5", 18, "B"), ("GEN", "+9", "+26a", None, "F"),
        ("FC", None, None, 24, "D")]

    resource = exporter.parse_resource(lines(
        "* Gain 10 Kazama Essence on normal hit and 7 on block or airborne hit"), "Kazama Essence")
    assert (resource["resource_gain_hit"], resource["resource_gain_block"], resource["resource_gain_airborne_hit"]) == (10, 7, 7)
    assert exporter.parse_resource(lines("* Gain 10 Kazama Essence on Heat activation"),
                                   "Kazama Essence")["resource_gain_heat_activation"] == 10
    assert exporter.parse_resource(lines("* Gain 20 Kazama Essence"), "Kazama Essence")["resource_gain_start"] == 20
    generic = exporter.parse_resource(lines("* Gain 8 Kazama Essence on hit"), "Kazama Essence")
    assert (generic["resource_gain_hit"], generic["resource_gain_airborne_hit"], generic["resource_gain_block"]) == (8, 8, "")
    assert exporter.parse_resource(lines("* Gain 8 Kazama Essence on hit"), "")["resource_gain_hit"] == ""

    install = exporter.parse_install(lines(
        "* DA: +9 damage on hit (30)\n* Divine Aura: 12 chip damage on block\n* DA: Range increases to 4.0"),
        ["DA", "Divine Aura"])
    assert install == {"install_damage_bonus": 9, "install_chip": 12, "install_range": 4.0}
    assert exporter.parse_install(lines("* Deals 8 (DA:11) chip damage on block"), ["DA"])["install_chip"] == 11

    assert exporter.parse_parry_levels(lines("* Parries low punches or kicks\n* Parries throws")) == "low|throw"
    assert exporter.parse_parry_levels(lines("* Sabaki, parries mid or high punches or kicks")) == "high|mid"
    assert exporter.parse_parry_levels(lines("* Punch sabaki\n* Parry state 4~15")) == exporter.UNKNOWN



def test_full_combat_binding_exporter_parses_skipped_mechanics():
    exporter = _exporter()
    lines = exporter.note_lines

    throw, used = exporter.parse_attack_throw(lines("* Transition to attack throw on front standing or airborne hit"))
    assert (throw["attack_throw"], throw["attack_throw_front_only"], throw["attack_throw_standing_only"],
            throw["attack_throw_airborne"]) == ("hit", 1, 1, 1) and len(used) == 1
    counter, _ = exporter.parse_attack_throw(lines("* Transition to attack throw on CH, +22 damage, total 44"))
    assert (counter["attack_throw"], counter["attack_throw_damage"]) == ("counter_hit", 22)
    standing, _ = exporter.parse_attack_throw(lines("* Transition to attack throw on standing front hit"))
    assert (standing["attack_throw_standing_only"], standing["attack_throw_airborne"]) == (1, 0)
    timed, used = exporter.parse_attack_throw(lines("* Transition to attack throw after 2nd hit"))
    assert timed["attack_throw"] == "" and used == []

    assert exporter.parse_back_turned_hit(lines("* Hit vs BT +12a (+2)")) == {
        "back_turned_hit_advantage": 12, "back_turned_hit_effect": "launch"}
    assert exporter.parse_back_turned_hit(lines("* +10a (+1) and Balcony Break on BT hit"))["back_turned_hit_advantage"] == 10

    rules = exporter.parse_damage_rules(lines(
        "* Deals 5 recoverable damage\n* Cannot cause a K.O.\n* Power up in Heat (ps5~12)"), "Inner Peace", "f+1+2")
    assert (rules["recoverable_damage"], rules["cannot_ko"], rules["heat_parry"]) == (5, 1, "5~12")
    assert exporter.parse_damage_rules(lines("* Damage increases with lower health, maximum 82"), "", "")[
        "rage_art_max_damage"] == 82
    assert exporter.parse_damage_rules([], "Ki Charge", "1+2+3+4")["ki_charge"] == 1

    assert exporter.damage_parts("[12;12]", [12.0, 12.0]) == [12.0]
    assert exporter.damage_parts("15,15,15", [15.0, 15.0, 15.0]) == [15.0, 15.0, 15.0]
    assert exporter.result_state("r20 BT", {"GEN"}) == ("BT", 0)

def test_full_combat_bindings_link_jun_state():
    import csv

    bindings = REPO_ROOT / "data" / "generated" / "full_combat_bindings.csv"
    rows = {row["stable_id"]: row for row in csv.DictReader(bindings.open(encoding="utf-8"))
            if row["character"] == "jun"}
    assert rows["jun:7"]["result_stance"] == "IZU"
    assert rows["jun:65"]["parry_outcomes"] == "jun:147|jun:147|jun:147|jun:147"
    assert rows["jun:67~GEN"]["variant_of"] == "jun:67" and rows["jun:67~GEN"]["recovery"] == "18"
    assert rows["jun:2"]["heat_cost_frames"] == "450"
    stances = list(csv.DictReader((bindings.parent / "full_combat_stances.csv").open(encoding="utf-8")))
    gen = next(row for row in stances if row["character"] == "jun" and row["stance"] == "GEN")
    assert (gen["can_guard"], gen["auto_parry"], gen["parry_outcome_low"], gen["parry_outcome_throw"]) == (
        "0", "low|throw", "jun:142", "jun:143")



def test_full_combat_routes_resolve_to_stable_ids():
    import csv

    routes = REPO_ROOT / "data" / "generated" / "full_combat_routes.csv"
    rows = [row for row in csv.DictReader(routes.open(encoding="utf-8")) if row["character"] == "jun"]
    assert {row["category"] for row in rows} == {"midscreen", "wall", "heat", "counter_hit"}
    assert all(row["input"].startswith(("jun:", "@")) for row in rows)
    midscreen = [row["input"] for row in rows if row["route"] == "df2_beginner"]
    assert midscreen == ["jun:43", "jun:20", "jun:7", "jun:124", "jun:30", "jun:124"]


def test_route_export_rejects_unknown_commands(tmp_path):
    import shutil

    exporter = _exporter()
    data_root = tmp_path / "data"
    shutil.copytree(REPO_ROOT / "data" / "characters", data_root / "characters")
    module = data_root / "character_modules" / "jun"
    module.mkdir(parents=True)
    (module / "routes.yaml").write_text(
        "routes:\n  - name: bad\n    category: midscreen\n    steps: [\"df+2\", \"not-a-move\"]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not in the catalog"):
        exporter.export(REPO_ROOT / "data" / "generated" / "full_move_catalog.json", data_root,
                        tmp_path / "out" / "bindings.csv", ["jun"])

def test_heat_and_rage_requirements_come_from_command_prefixes():
    catalog = load_compiled_catalog(REPO_ROOT / "data" / "generated" / "full_move_catalog.json")
    jun = {move["stable_id"]: move for move in catalog.moves if move["stable_id"].startswith("jun:")}
    # Heat Engager, Heat Burst, and Heat Smash roles; only H./R. prefixes require Heat or Rage.
    assert jun["jun:21"]["mechanics"]["heat_engager"] and not jun["jun:21"]["legal_state"]["requires_heat"]
    assert jun["jun:1"]["mechanics"]["heat_burst"] and not jun["jun:1"]["legal_state"]["requires_heat"]
    assert jun["jun:4"]["mechanics"]["heat_smash"] and jun["jun:4"]["legal_state"]["requires_heat"]
    assert jun["jun:5"]["legal_state"]["requires_rage"]
    for move in catalog.moves:
        requirements = move["mechanics"]["requirements"]
        assert move["legal_state"]["requires_heat"] == ("H" in requirements)
        assert move["legal_state"]["requires_rage"] == ("R" in requirements)

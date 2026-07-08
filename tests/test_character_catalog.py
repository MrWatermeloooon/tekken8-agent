from t8_agent.data.character_catalog import load_character_catalog
import yaml


def test_jun_character_catalog_loads() -> None:
    catalog = load_character_catalog("data/characters/jun.yaml")

    assert catalog.character_id == "jun"
    assert len(catalog.moves) == 149
    assert len(catalog.core_moves) >= 15
    assert len(catalog.combos) >= 3


def test_jun_combos_reference_known_moves() -> None:
    catalog = load_character_catalog("data/characters/jun.yaml")
    move_ids = {move.move_id for move in catalog.moves}

    for combo in catalog.combos:
        assert combo.starter in move_ids
        assert set(combo.route).issubset(move_ids)


def test_jun_catalog_includes_system_and_throw_moves() -> None:
    catalog = load_character_catalog("data/characters/jun.yaml")
    commands = {move.command for move in catalog.moves}

    assert "2+3" in commands
    assert "H.2+3" in commands
    assert "R.df+1+2" in commands
    assert "Left throw" in commands
    assert "Right throw" in commands


def test_universal_actions_cover_movement_defense_and_systems() -> None:
    data = yaml.safe_load(open("data/universal_actions.yaml", encoding="utf-8"))
    action_ids = {item["id"] for item in data["actions"]}

    assert {"sidestep_left", "sidestep_right", "block_high", "block_low"}.issubset(action_ids)
    assert {"low_parry", "throw_break_1", "throw_break_2", "throw_break_1p2"}.issubset(action_ids)
    assert {"heat_burst", "heat_smash", "rage_art"}.issubset(action_ids)

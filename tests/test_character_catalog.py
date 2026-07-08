from t8_agent.data.character_catalog import load_character_catalog


def test_jun_character_catalog_loads() -> None:
    catalog = load_character_catalog("data/characters/jun.yaml")

    assert catalog.character_id == "jun"
    assert len(catalog.moves) >= 20
    assert len(catalog.core_moves) >= 15
    assert len(catalog.combos) >= 3


def test_jun_combos_reference_known_moves() -> None:
    catalog = load_character_catalog("data/characters/jun.yaml")
    move_ids = {move.move_id for move in catalog.moves}

    for combo in catalog.combos:
        assert combo.starter in move_ids
        assert set(combo.route).issubset(move_ids)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from t8_agent.roster.catalog import PROBABILITY_FIELDS, load_catalog
from t8_agent.roster.evaluation import MatchupEvaluation
from t8_agent.roster.scheduler import CurriculumStage, LeagueEntry, MatchupScheduler
from t8_agent.roster.temporal import MatchupObservationEncoder, TemporalFrame


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_roster_has_exact_style_matrix_and_generated_files() -> None:
    catalog = load_catalog(REPO_ROOT / "data")
    assert len(catalog.characters) == 42
    assert len(catalog.archetypes) == 10
    assert len(catalog.variations) == 5
    assert len(catalog.profiles) == 2100
    assert {len(catalog.profiles_for(character.id)) for character in catalog.characters} == {50}
    assert [profile.id for profile in catalog.profiles] == list(range(2100))
    assert len({profile.name for profile in catalog.profiles}) == 2100
    manifest = json.loads((REPO_ROOT / "data/generated/manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_profiles"] == 2100
    assert manifest["frame_data_characters"] == 41
    assert manifest["character_move_specs"] == 252
    move_lines = (REPO_ROOT / "data/generated/character_move_specs.csv").read_text(encoding="utf-8").splitlines()
    assert len(move_lines) == 253
    for character in catalog.characters:
        module = REPO_ROOT / "data/character_modules" / character.slug / "matchup.yaml"
        assert module.is_file()


def test_profiles_are_parameterized_distributions() -> None:
    catalog = load_catalog(REPO_ROOT / "data")
    profile = catalog.profiles_for("reina", "rushdown")[2]
    first = profile.episode_variant(123, 9)
    repeated = profile.episode_variant(123, 9)
    other = profile.episode_variant(123, 10)
    assert first == repeated
    assert first != other
    for field in PROBABILITY_FIELDS:
        assert 0.0 <= getattr(first, field) <= 1.0
    assert first.reaction_min <= first.reaction_max
    assert len({catalog.profiles_for(slug, "rushdown")[0].aggression for slug in ("king", "reina", "jack-8")}) == 3


def test_temporal_matchup_observation_contains_identity_style_and_history() -> None:
    encoder = MatchupObservationEncoder(base_size=13, history_length=8)
    base = np.zeros(13, dtype=np.float32)
    first = encoder.encode(
        base,
        opponent_character_id=31,
        opponent_archetype_id=6,
        frame=TemporalFrame(move_id=20, animation_phase=0.4, stance_id=2, hit_level=2,
                            delay_frames=7, outcome=-1.0, distance=1.8, side_movement=0.3),
    )
    second = encoder.encode(
        base,
        opponent_character_id=31,
        opponent_archetype_id=6,
        frame=TemporalFrame(move_id=21, animation_phase=0.7, stance_id=3, hit_level=3,
                            delay_frames=11, outcome=1.0, distance=1.2, side_movement=-0.2),
    )
    assert encoder.observation_size == 95
    assert first.shape == (95,)
    assert second.shape == (95,)
    assert np.count_nonzero(first[13:21]) == 8
    assert first[21 + 6] == 1.0
    assert not np.array_equal(first[-16:], second[-16:])
    encoder.reset()
    reset = encoder.encode(base, opponent_character_id=30, opponent_archetype_id=6, frame=TemporalFrame())
    assert not np.array_equal(first[13:21], reset[13:21])


def test_curriculum_and_weakness_scheduler_cover_all_four_stages() -> None:
    catalog = load_catalog(REPO_ROOT / "data")
    scheduler = MatchupScheduler(catalog, seed=7)
    stage1 = scheduler.candidates()
    assert stage1
    assert all("fundamentals" in catalog.characters[value.character_id].groups for value in stage1)
    with pytest.raises(ValueError):
        scheduler.set_stage(CurriculumStage.CHARACTER_GROUPS)
    scheduler.set_stage(CurriculumStage.CHARACTER_GROUPS, groups=["grappler"])
    assert {catalog.characters[value.character_id].slug for value in scheduler.candidates()} == {"king", "armor-king"}
    scheduler.set_stage(CurriculumStage.FULL_ROSTER)
    assert len(scheduler.candidates()) == 2100
    weak = catalog.profiles[0]
    dominant = catalog.profiles[1]
    scheduler.record(weak, wins=2, losses=8)
    scheduler.record(dominant, wins=98, losses=2)
    assert scheduler.priority(weak) > scheduler.priority(dominant)
    scheduler.register_league_entry(LeagueEntry("history-10", "historical_checkpoint", "checkpoints/update_10.t8ppo"))
    scheduler.register_league_entry(LeagueEntry("exploit-low", "exploit_policy", "league/low.t8ppo", exploit_severity=0.9))
    scheduler.register_league_entry(LeagueEntry("human-wall", "human_failure", "failures/wall.jsonl", exploit_severity=0.7))
    scheduler.set_stage(CurriculumStage.ADVERSARIAL_LEAGUE)
    assert len(scheduler.candidates()) == 2103
    assert len(scheduler.sample(32)) == 32


def test_matchup_evaluation_tracks_required_rates_elo_and_forgetting(tmp_path: Path) -> None:
    catalog = load_catalog(REPO_ROOT / "data")
    profile = catalog.profiles_for("king", "throw_heavy")[0]
    evaluation = MatchupEvaluation(catalog, forgetting_threshold=0.10)
    cell = evaluation.record_batch(
        profile, wins=8, losses=2,
        punishment=(7, 9), throw_break=(5, 8), low_defence=(4, 7),
        string_interrupt=(3, 5), sidestep=(6, 9), heat_defence=(4, 6), wall_escape=(2, 4),
    )
    assert cell.win_rate == pytest.approx(0.8)
    assert cell.elo > 1500.0
    evaluation.record_batch(profile, wins=0, losses=10)
    assert cell.catastrophic_forgetting
    payload = evaluation.to_dict()
    detail = payload["details"][0]
    assert detail["throw_break_accuracy"] == pytest.approx(5 / 8)
    assert detail["heat_defence"] == pytest.approx(4 / 6)
    json_path = tmp_path / "matchups.json"
    csv_path = tmp_path / "matchups.csv"
    evaluation.write_json(json_path)
    evaluation.write_csv(csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["matrix"]
    assert "throw_heavy" in csv_path.read_text(encoding="utf-8").splitlines()[0]

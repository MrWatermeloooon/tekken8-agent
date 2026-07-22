"""Data-driven full-roster opponent, curriculum, and evaluation support."""

from .catalog import Archetype, Character, OpponentCatalog, OpponentProfile, load_catalog
from .evaluation import MatchupEvaluation, MatchupMetrics
from .scheduler import CurriculumStage, LeagueEntry, MatchupScheduler, MatchupStats
from .temporal import MatchupObservationEncoder, TemporalFrame

__all__ = [
    "Archetype", "Character", "CurriculumStage", "MatchupEvaluation",
    "LeagueEntry", "MatchupMetrics", "MatchupObservationEncoder", "MatchupScheduler",
    "MatchupStats", "OpponentCatalog", "OpponentProfile", "TemporalFrame",
    "load_catalog",
]

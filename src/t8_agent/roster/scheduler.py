from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import random
from typing import Iterable, Literal

from .catalog import OpponentCatalog, OpponentProfile


class CurriculumStage(IntEnum):
    JUN_FUNDAMENTALS = 1
    CHARACTER_GROUPS = 2
    FULL_ROSTER = 3
    ADVERSARIAL_LEAGUE = 4


LeagueSource = Literal["historical_checkpoint", "exploit_policy", "human_failure"]


@dataclass(frozen=True)
class LeagueEntry:
    id: str
    source: LeagueSource
    path: str
    character_id: int | None = None
    archetype_id: int | None = None
    exploit_severity: float = 0.0


@dataclass
class MatchupStats:
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    best_win_rate: float = 0.0
    recent_win_rate: float = 0.5
    exploit_severity: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.episodes == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.episodes

    @property
    def uncertainty(self) -> float:
        return 1.0 / math.sqrt(self.episodes + 1.0)

    @property
    def regression(self) -> float:
        return max(0.0, self.best_win_rate - self.recent_win_rate)


ScheduledOpponent = OpponentProfile | LeagueEntry


class MatchupScheduler:
    """Staged weakness scheduler with league and anti-forgetting support."""

    def __init__(self, catalog: OpponentCatalog, *, seed: int = 2027) -> None:
        self.catalog = catalog
        self.stage = CurriculumStage.JUN_FUNDAMENTALS
        self.active_groups: tuple[str, ...] = ()
        self.stats: dict[str, MatchupStats] = {}
        self.league: list[LeagueEntry] = []
        self._rng = random.Random(seed)

    def set_stage(self, stage: CurriculumStage, *, groups: Iterable[str] = ()) -> None:
        self.stage = CurriculumStage(stage)
        self.active_groups = tuple(dict.fromkeys(str(group) for group in groups))
        if self.stage == CurriculumStage.CHARACTER_GROUPS and not self.active_groups:
            raise ValueError("character-group curriculum requires at least one active group")

    def register_league_entry(self, entry: LeagueEntry) -> None:
        if any(existing.id == entry.id for existing in self.league):
            raise ValueError(f"duplicate league entry: {entry.id}")
        self.league.append(entry)
        self.stats[self._key(entry)] = MatchupStats(exploit_severity=max(0.0, entry.exploit_severity))

    def record(
        self,
        opponent: ScheduledOpponent,
        *,
        wins: int,
        losses: int,
        draws: int = 0,
        recent_win_rate: float | None = None,
        exploit_severity: float | None = None,
    ) -> None:
        if min(wins, losses, draws) < 0 or wins + losses + draws == 0:
            raise ValueError("record requires a non-empty, non-negative outcome batch")
        stats = self.stats.setdefault(self._key(opponent), MatchupStats())
        stats.episodes += wins + losses + draws
        stats.wins += wins
        stats.losses += losses
        stats.draws += draws
        batch_rate = (wins + 0.5 * draws) / (wins + losses + draws)
        stats.recent_win_rate = batch_rate if recent_win_rate is None else float(recent_win_rate)
        stats.best_win_rate = max(stats.best_win_rate, stats.recent_win_rate)
        if exploit_severity is not None:
            stats.exploit_severity = max(0.0, float(exploit_severity))

    def priority(self, opponent: ScheduledOpponent) -> float:
        stats = self.stats.setdefault(self._key(opponent), MatchupStats())
        weakness = 1.0 - stats.win_rate
        value = weakness + stats.uncertainty + stats.regression + stats.exploit_severity
        if isinstance(opponent, OpponentProfile) and stats.win_rate < 0.25:
            value *= 1.35 if opponent.variation_id <= 1 else 0.65
        elif stats.win_rate > 0.95:
            value *= 0.10
        return max(value, 1e-6)

    def candidates(self) -> tuple[ScheduledOpponent, ...]:
        profiles = self.catalog.profiles
        if self.stage == CurriculumStage.JUN_FUNDAMENTALS:
            profiles = tuple(
                profile for profile in profiles
                if "fundamentals" in self.catalog.characters[profile.character_id].groups
                and profile.archetype_id in {0, 1, 2, 3, 4}
                and profile.variation_id <= 1
            )
        elif self.stage == CurriculumStage.CHARACTER_GROUPS:
            groups = set(self.active_groups)
            profiles = tuple(
                profile for profile in profiles
                if groups.intersection(self.catalog.characters[profile.character_id].groups)
            )
        if self.stage == CurriculumStage.ADVERSARIAL_LEAGUE:
            return (*profiles, *self.league)
        return tuple(profiles)

    def sample(self, count: int) -> list[ScheduledOpponent]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        candidates = self.candidates()
        if not candidates:
            raise RuntimeError("curriculum has no eligible opponents")
        weights = [self.priority(opponent) for opponent in candidates]
        return self._rng.choices(candidates, weights=weights, k=count)

    @staticmethod
    def _key(opponent: ScheduledOpponent) -> str:
        return f"profile:{opponent.id}" if isinstance(opponent, OpponentProfile) else f"league:{opponent.id}"

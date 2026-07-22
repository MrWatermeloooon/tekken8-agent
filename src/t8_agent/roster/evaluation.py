from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

from .catalog import OpponentCatalog, OpponentProfile


@dataclass
class MatchupMetrics:
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    punishment_attempts: int = 0
    punishment_successes: int = 0
    throw_break_attempts: int = 0
    throw_break_successes: int = 0
    low_defence_attempts: int = 0
    low_defence_successes: int = 0
    string_interrupt_attempts: int = 0
    string_interrupt_successes: int = 0
    sidestep_attempts: int = 0
    sidestep_successes: int = 0
    heat_defence_attempts: int = 0
    heat_defence_successes: int = 0
    wall_escape_attempts: int = 0
    wall_escape_successes: int = 0
    elo: float = 1500.0
    best_win_rate: float = 0.0
    previous_win_rate: float = 0.0
    catastrophic_forgetting: bool = False

    @property
    def win_rate(self) -> float:
        return 0.0 if self.episodes == 0 else self.wins / self.episodes

    @property
    def score_rate(self) -> float:
        return 0.5 if self.episodes == 0 else (self.wins + 0.5 * self.draws) / self.episodes

    def rate(self, successes: str, attempts: str) -> float | None:
        denominator = getattr(self, attempts)
        return None if denominator == 0 else getattr(self, successes) / denominator


class MatchupEvaluation:
    """Per-character/archetype outcomes, behavior rates, Elo, and regressions."""

    def __init__(self, catalog: OpponentCatalog, *, forgetting_threshold: float = 0.10) -> None:
        self.catalog = catalog
        self.forgetting_threshold = forgetting_threshold
        self.cells: dict[tuple[int, int], MatchupMetrics] = {}

    def record_batch(
        self,
        profile: OpponentProfile,
        *,
        wins: int,
        losses: int,
        draws: int = 0,
        punishment: tuple[int, int] = (0, 0),
        throw_break: tuple[int, int] = (0, 0),
        low_defence: tuple[int, int] = (0, 0),
        string_interrupt: tuple[int, int] = (0, 0),
        sidestep: tuple[int, int] = (0, 0),
        heat_defence: tuple[int, int] = (0, 0),
        wall_escape: tuple[int, int] = (0, 0),
    ) -> MatchupMetrics:
        total = wins + losses + draws
        if min(wins, losses, draws) < 0 or total <= 0:
            raise ValueError("evaluation batch must contain non-negative outcomes")
        cell = self.cells.setdefault((profile.character_id, profile.archetype_id), MatchupMetrics())
        prior_rate = cell.win_rate
        cell.previous_win_rate = prior_rate
        cell.episodes += total
        cell.wins += wins
        cell.losses += losses
        cell.draws += draws
        for value, prefix in (
            (punishment, "punishment"), (throw_break, "throw_break"),
            (low_defence, "low_defence"), (string_interrupt, "string_interrupt"),
            (sidestep, "sidestep"), (heat_defence, "heat_defence"),
            (wall_escape, "wall_escape"),
        ):
            successes, attempts = value
            if not 0 <= successes <= attempts:
                raise ValueError(f"{prefix} successes must be between zero and attempts")
            setattr(cell, f"{prefix}_successes", getattr(cell, f"{prefix}_successes") + successes)
            setattr(cell, f"{prefix}_attempts", getattr(cell, f"{prefix}_attempts") + attempts)
        batch_score = (wins + 0.5 * draws) / total
        expected = 1.0 / (1.0 + 10.0 ** ((1500.0 - cell.elo) / 400.0))
        cell.elo += 24.0 * (batch_score - expected)
        cell.catastrophic_forgetting = (
            cell.best_win_rate > 0.0 and
            cell.score_rate < cell.best_win_rate - self.forgetting_threshold
        )
        cell.best_win_rate = max(cell.best_win_rate, cell.score_rate)
        return cell

    def matrix(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for character in self.catalog.characters:
            cells = [self.cells.get((character.id, archetype.id), MatchupMetrics()) for archetype in self.catalog.archetypes]
            episodes = sum(cell.episodes for cell in cells)
            wins = sum(cell.wins for cell in cells)
            rows.append({
                "character_id": character.id,
                "character": character.slug,
                "archetypes": {archetype.key: cells[archetype.id].win_rate for archetype in self.catalog.archetypes},
                "overall": 0.0 if episodes == 0 else wins / episodes,
                "episodes": episodes,
                "matchup_elo": 1500.0 if not cells else sum(cell.elo for cell in cells) / len(cells),
                "catastrophic_forgetting": any(cell.catastrophic_forgetting for cell in cells),
            })
        return rows

    def to_dict(self) -> dict[str, Any]:
        details: list[dict[str, Any]] = []
        for (character_id, archetype_id), cell in sorted(self.cells.items()):
            row = asdict(cell)
            row.update({
                "character": self.catalog.characters[character_id].slug,
                "archetype": self.catalog.archetypes[archetype_id].key,
                "win_rate": cell.win_rate,
                "score_rate": cell.score_rate,
                "punishment_accuracy": cell.rate("punishment_successes", "punishment_attempts"),
                "throw_break_accuracy": cell.rate("throw_break_successes", "throw_break_attempts"),
                "low_defence": cell.rate("low_defence_successes", "low_defence_attempts"),
                "string_interruption": cell.rate("string_interrupt_successes", "string_interrupt_attempts"),
                "sidestep_success": cell.rate("sidestep_successes", "sidestep_attempts"),
                "heat_defence": cell.rate("heat_defence_successes", "heat_defence_attempts"),
                "wall_escape": cell.rate("wall_escape_successes", "wall_escape_attempts"),
            })
            details.append(row)
        return {"roster_as_of": self.catalog.as_of, "matrix": self.matrix(), "details": details}

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self.matrix()
        fieldnames = ["character", *[value.key for value in self.catalog.archetypes], "overall", "episodes", "matchup_elo", "catastrophic_forgetting"]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "character": row["character"], **row["archetypes"],
                    "overall": row["overall"], "episodes": row["episodes"],
                    "matchup_elo": row["matchup_elo"],
                    "catastrophic_forgetting": row["catastrophic_forgetting"],
                })

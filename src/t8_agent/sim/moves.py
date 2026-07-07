from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HitLevel(str, Enum):
    HIGH = "high"
    MID = "mid"
    LOW = "low"
    THROW = "throw"


@dataclass(frozen=True)
class MoveSpec:
    name: str
    command: str
    hit_level: HitLevel
    startup: int
    active: int
    recovery: int
    damage: float
    range: float
    hitstun: int
    blockstun: int
    pushback: float
    whiff_recovery: int = 0
    launches: bool = False

    @property
    def total_frames(self) -> int:
        return self.startup + self.active + self.recovery + self.whiff_recovery


JUN_MOVES: dict[str, MoveSpec] = {
    "jab": MoveSpec(
        name="Jun jab",
        command="1",
        hit_level=HitLevel.HIGH,
        startup=10,
        active=2,
        recovery=13,
        damage=7.0,
        range=0.82,
        hitstun=14,
        blockstun=7,
        pushback=0.08,
    ),
    "df1": MoveSpec(
        name="Jun df1",
        command="df+1",
        hit_level=HitLevel.MID,
        startup=13,
        active=3,
        recovery=18,
        damage=12.0,
        range=0.95,
        hitstun=18,
        blockstun=11,
        pushback=0.11,
    ),
    "f2": MoveSpec(
        name="Jun f2",
        command="f+2",
        hit_level=HitLevel.MID,
        startup=16,
        active=3,
        recovery=22,
        damage=18.0,
        range=1.25,
        hitstun=24,
        blockstun=14,
        pushback=0.16,
    ),
    "db3": MoveSpec(
        name="Jun low poke",
        command="db+3",
        hit_level=HitLevel.LOW,
        startup=18,
        active=3,
        recovery=24,
        damage=10.0,
        range=0.9,
        hitstun=17,
        blockstun=15,
        pushback=0.06,
    ),
    "hopkick": MoveSpec(
        name="Jun hopkick",
        command="uf+4",
        hit_level=HitLevel.MID,
        startup=15,
        active=4,
        recovery=30,
        damage=21.0,
        range=0.88,
        hitstun=34,
        blockstun=18,
        pushback=0.18,
        launches=True,
    ),
    "throw": MoveSpec(
        name="Jun throw",
        command="1+3",
        hit_level=HitLevel.THROW,
        startup=12,
        active=2,
        recovery=24,
        damage=25.0,
        range=0.48,
        hitstun=30,
        blockstun=0,
        pushback=0.22,
    ),
}


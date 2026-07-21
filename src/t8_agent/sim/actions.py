from __future__ import annotations

from enum import Enum


class SimAction(str, Enum):
    """Live controller action names in the frozen native V2 action order."""

    NEUTRAL = "neutral"
    WALK_FORWARD = "walk_forward"
    WALK_BACK = "walk_back"
    DASH_FORWARD = "dash_forward"
    DASH_BACK = "dash_back"
    CROUCH = "crouch"
    STAND = "stand"
    JUMP = "jump"
    SIDESTEP_LEFT = "sidestep_left"
    SIDESTEP_RIGHT = "sidestep_right"
    SIDEWALK_LEFT = "sidewalk_left"
    SIDEWALK_RIGHT = "sidewalk_right"
    BLOCK_HIGH = "block_high"
    BLOCK_LOW = "block_low"
    LOW_PARRY = "low_parry"
    THROW_BREAK_1 = "throw_break_1"
    THROW_BREAK_2 = "throw_break_2"
    THROW_BREAK_1_2 = "throw_break_1p2"
    JAB = "jab"
    DF1 = "df1"
    F2 = "f2"
    DB3 = "db3"
    HOPKICK = "hopkick"
    THROW = "throw"
    HEAT_BURST = "heat_burst"
    HEAT_SMASH = "heat_smash"
    RAGE_ART = "rage_art"

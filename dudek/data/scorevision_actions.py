"""
Bittensor subnet 44 (Score / Turbovision private track) action vocabulary.

Source of truth matches score-technologies/turbovision:
https://github.com/score-technologies/turbovision/blob/main/scorevision/utils/actions.py
Miner responses use snake_case strings, e.g. FramePrediction(frame=45, action="pass").
"""

from __future__ import annotations

import enum
from typing import Final


class ScorevisionAction(str, enum.Enum):
    """Order matches ACTION_CLASS_INDEX in upstream scorevision/utils/actions.py."""

    PASS = "pass"
    PASS_RECEIVED = "pass_received"
    RECOVERY = "recovery"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    BALL_OUT_OF_PLAY = "ball_out_of_play"
    CLEARANCE = "clearance"
    TAKE_ON = "take_on"
    SUBSTITUTION = "substitution"
    BLOCK = "block"
    AERIAL_DUEL = "aerial_duel"
    SHOT = "shot"
    SAVE = "save"
    FOUL = "foul"
    GOAL = "goal"


NUM_SCOREVISION_ACTIONS: Final[int] = len(ScorevisionAction)

ACTION_CLASS_INDEX: Final[dict[str, int]] = {
    action.value: idx for idx, action in enumerate(ScorevisionAction)
}

"""
Map SoccerNet Ball Action Spotting (BAS) labels onto Score Vision subnet-44 actions.

BAS has 12 coarse classes; Score Vision has 16. There is no perfect 1:1 mapping.
Events that cannot be expressed without guessing are omitted (return None) so they
are not used as supervision. For production mining quality, prefer labels that
already use the Score Vision ontology (extend Annotation loading separately).
"""

from __future__ import annotations

from typing import Any, Optional

from dudek.data.scorevision_actions import ScorevisionAction
from dudek.data.team_bas import BASLabel


def ball_labels_json_to_scorevision(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of SoccerNet-style label JSON with BAS labels replaced by Score Vision strings."""

    out_rows: list[dict[str, Any]] = []
    for row in data.get("annotations", []):
        raw = row.get("label")
        if raw is None:
            continue
        try:
            bas = BASLabel(raw)
        except ValueError:
            continue
        sv = bas_label_to_scorevision_action(bas)
        if sv is None:
            continue
        new_row = dict(row)
        new_row["label"] = sv.value
        out_rows.append(new_row)
    return {"annotations": out_rows}


def bas_label_to_scorevision_action(label: BASLabel) -> Optional[ScorevisionAction]:
    """Best-effort mapping from SN-BAS BASLabel to miner-valid ScorevisionAction."""

    return _BAS_TO_SCORE.get(label)


_BAS_TO_SCORE: dict[BASLabel, Optional[ScorevisionAction]] = {
    BASLabel.PASS: ScorevisionAction.PASS,
    BASLabel.DRIVE: ScorevisionAction.TAKE_ON,
    BASLabel.HEADER: ScorevisionAction.AERIAL_DUEL,
    BASLabel.HIGH_PASS: ScorevisionAction.CLEARANCE,
    BASLabel.OUT: ScorevisionAction.BALL_OUT_OF_PLAY,
    BASLabel.CROSS: ScorevisionAction.PASS,
    BASLabel.THROW_IN: ScorevisionAction.BALL_OUT_OF_PLAY,
    BASLabel.SHOT: ScorevisionAction.SHOT,
    BASLabel.BALL_PLAYER_BLOCK: ScorevisionAction.BLOCK,
    BASLabel.PLAYER_SUCCESSFUL_TACKLE: ScorevisionAction.TACKLE,
    BASLabel.FREE_KICK: ScorevisionAction.PASS,
    BASLabel.GOAL: ScorevisionAction.GOAL,
}

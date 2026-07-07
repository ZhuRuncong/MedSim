"""Evidence-based scoring (PLAN §7).

`apply_points` is the single choke-point every agent uses to change the score.
It records a fully traceable :class:`ScoreEvent` (action, delta, agent, reason,
guideline citation) — the same information LangSmith would capture — so every
point is auditable.
"""
from __future__ import annotations

from typing import Optional

from . import config
from .state import GameState, ScoreEvent


def apply_points(
    state: GameState,
    action: str,
    agent: str,
    reason: str = "",
    citation: Optional[str] = None,
    delta: Optional[int] = None,
    scale: float = 1.0,
) -> int:
    """Apply a rubric action to ``state`` and return the point delta.

    ``action`` should be a key in :data:`config.POINTS`. Pass ``delta`` to
    override the rubric value (e.g. partial credit), or ``scale`` to weight it
    (e.g. by case severity — missing a critical diagnosis costs more).
    """
    if delta is None:
        delta = round(config.POINTS.get(action, 0) * scale)

    state.points += delta
    state.level = config.level_for_points(state.points)
    state.score_log.append(
        ScoreEvent(
            action=action,
            delta=delta,
            agent=agent,
            reason=reason,
            citation=citation,
            turn=state.turn,
        )
    )
    return delta

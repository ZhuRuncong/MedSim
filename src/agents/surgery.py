"""SurgeryAgent (PLAN §5): emergency-surgery requests.

Calls the emergency_surgery tool to validate indication and adjusts points
(positive if indicated, strongly negative if not).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .. import safety
from ..scoring import apply_points
from ..state import GameState, Message
from ..tools import evaluate_surgery

AGENT = "SurgeryAgent"


def request_surgery(state: GameState, disease: dict, procedure: str,
                    rng: np.random.Generator) -> List[Message]:
    ev = evaluate_surgery(procedure, disease, rng)
    state.surgery_requested = procedure
    state.surgeries.append(procedure)
    citation = disease.get("guideline", {}).get("url")

    if ev["indicated"]:
        delta = apply_points(state, "emergency_surgery_indicated", AGENT,
                             reason=ev["rationale"], citation=citation)
        kind = "gain"
    else:
        delta = apply_points(state, "unwarranted_surgery", AGENT,
                             scale=safety.severity_multiplier(disease),
                             reason=ev["rationale"], citation=citation)
        kind = "fail"

    text = f"Requested {procedure}. {ev['outcome']}"
    return [Message(role="surgery", text=text, kind=kind, points_delta=delta,
                    citations=ev.get("citations", []), turn=state.turn)]

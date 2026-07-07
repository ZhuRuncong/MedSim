"""SupervisorAgent (PLAN §5): the central router.

Every student action passes through here. The supervisor advances the turn
counter, enforces the case lifecycle (blocks actions on a finished case),
dispatches to the specialised agent, and applies small history/exam rewards.
It mirrors the LangGraph edges in `graph.py`.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..scoring import apply_points
from ..state import GameState, Message
from ..util import normalize
from . import clinical, critic, knowledge, patient, surgery

AGENT = "SupervisorAgent"

ACTION_TYPES = {
    "ask_history", "perform_exam", "order_test", "prescribe_drug",
    "request_surgery", "knowledge_query", "submit_diagnosis",
}


# --------------------------------------------------------------------------- #
# Shared per-action handlers (reused by both the deterministic router below and
# the LangGraph nodes in graph.py, so the two paths stay exactly equivalent).
# --------------------------------------------------------------------------- #
def handle_history(state: GameState, disease: dict, question: str) -> List[Message]:
    q = (question or "").strip()
    if not q:
        return [Message(role="supervisor", text="Please enter a question.",
                        kind="info", turn=state.turn)]
    duplicate = normalize(q) in {normalize(x) for x in state.asked_history}
    state.asked_history.append(q)
    res = patient.answer_history(state, disease, q)
    if duplicate:  # no re-reward for re-asking the same question
        apply_points(state, "duplicate_action", AGENT,
                     reason="Repeated an earlier question.")
        return [Message(role="patient", text=res["answer"], kind="loss", turn=state.turn)]
    delta = (apply_points(state, "informative_history", AGENT)
             if res.get("informative") else 0)
    return [Message(role="patient", text=res["answer"], kind="info",
                    points_delta=delta, turn=state.turn,
                    llm=bool(res.get("llm")))]


def handle_exam(state: GameState, disease: dict, exam: str) -> List[Message]:
    exam = (exam or "").strip()
    if not exam:
        return [Message(role="supervisor", text="Please choose an exam.",
                        kind="info", turn=state.turn)]
    if exam in state.performed_exam:
        apply_points(state, "duplicate_action", AGENT)
        return [Message(role="supervisor", text=f"You already performed: {exam}.",
                        kind="loss", turn=state.turn)]
    state.performed_exam.append(exam)
    res = patient.perform_exam(state, disease, exam)
    delta = (apply_points(state, "informative_exam", AGENT)
             if res.get("informative") else 0)
    return [Message(role="patient", text=res["finding"], kind="info",
                    points_delta=delta, turn=state.turn)]


def route(state: GameState, disease: dict, action: dict,
          rng: np.random.Generator) -> List[Message]:
    atype = action.get("type")
    payload = action.get("payload")

    if state.status != "active":
        return [Message(role="supervisor",
                        text="This case is closed. Start a new case to continue.",
                        kind="info", turn=state.turn)]
    if atype not in ACTION_TYPES:
        return [Message(role="supervisor",
                        text=f"Unknown action '{atype}'.", kind="info", turn=state.turn)]

    state.turn += 1

    if atype == "ask_history":
        return handle_history(state, disease, str(payload or ""))

    if atype == "perform_exam":
        return handle_exam(state, disease, str(payload or ""))

    if atype == "order_test":
        return clinical.order_test(state, disease, str(payload), rng)

    if atype == "prescribe_drug":
        return clinical.prescribe_drug(state, disease, str(payload))

    if atype == "request_surgery":
        return surgery.request_surgery(state, disease, str(payload), rng)

    if atype == "knowledge_query":
        return knowledge.ask(state, str(payload))

    if atype == "submit_diagnosis":
        return critic.submit_diagnosis(state, disease, str(payload),
                                       differentials=action.get("differentials"),
                                       confidence=action.get("confidence"))

    return [Message(role="supervisor", text="No handler for that action.",
                    kind="info", turn=state.turn)]

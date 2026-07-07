"""CriticAgent (PLAN §5): evaluates the final diagnosis.

Compares the student's diagnosis against the hidden disease. Correct → large
bonus and case complete. A named differential → partial credit. Otherwise a
penalty and a retry; exhausting retries ends the case.

It also scores clinical METACOGNITION (all deterministic & auditable): credit
for listing the true diagnosis among ranked differentials, confidence
calibration (rewarding calibrated certainty and hedging, penalizing overconfident
errors), and a premature-closure penalty for committing with no work-up.
"""
from __future__ import annotations

from typing import List, Optional

from .. import safety
from ..scoring import apply_points
from ..state import GameState, Message
from ..util import normalize, words

AGENT = "CriticAgent"

_RANK_CREDIT = {1: 12, 2: 8, 3: 6}   # points for listing the true dx at rank N


def _is_correct(diagnosis: str, disease: dict) -> bool:
    """Whole-word (token-subset) match, not substring.

    A guess is correct when every word of the disease name — or of any alias —
    appears in the guess. This accepts "I think it's a heart attack" or a bare
    "MI", but rejects "migraine" (which merely *contains* the alias 'mi') and
    "acute" (which is merely *contained by* the name).
    """
    dtok = words(diagnosis)
    if not dtok:
        return False
    for name in [disease["name"], *disease.get("aliases", [])]:
        ntok = words(name)
        if ntok and ntok <= dtok:
            return True
    return False


def _is_partial(diagnosis: str, disease: dict) -> bool:
    dtok = words(diagnosis)
    if not dtok:  # guard: an empty guess must not match every differential
        return False
    for diff in disease.get("differentials", []):
        dftok = words(diff)
        if dftok and dftok <= dtok:
            return True
    return False


def _true_dx_rank(differentials: List[str], disease: dict) -> Optional[int]:
    """1-based rank of the true diagnosis within the student's ranked list."""
    for i, d in enumerate(differentials, 1):
        if _is_correct(d, disease):
            return i
    return None


def _calibration_events(state: GameState, correct: bool, confidence: Optional[int]) -> List[Message]:
    """Reward calibrated certainty / hedging, penalize overconfident errors."""
    if confidence is None:
        return []
    c = max(0, min(100, int(confidence))) / 100.0
    if correct and c >= 0.7:
        d = apply_points(state, "calibrated_confidence", AGENT,
                         reason=f"Correct at {confidence}% confidence (well-calibrated).")
        return [Message(role="critic", kind="gain", points_delta=d, turn=state.turn,
                        text=f"Well-calibrated — correct and {confidence}% confident.")]
    if not correct and c >= 0.7:
        d = apply_points(state, "overconfident_error", AGENT,
                         reason=f"Incorrect at {confidence}% confidence (overconfident).")
        return [Message(role="critic", kind="loss", points_delta=d, turn=state.turn,
                        text=f"Overconfident — {confidence}% sure but wrong. Watch for premature anchoring.")]
    if not correct and c <= 0.3:
        d = apply_points(state, "prudent_uncertainty", AGENT,
                         reason=f"Incorrect but flagged low confidence ({confidence}%).")
        return [Message(role="critic", kind="gain", points_delta=d, turn=state.turn,
                        text=f"Appropriate uncertainty — you flagged low confidence ({confidence}%).")]
    return []


def _reveal(disease: dict, turn: int) -> Message:
    g = disease.get("guideline", {})
    return Message(
        role="critic",
        text=(f"The correct diagnosis was **{disease['name']}**. "
              f"{disease.get('teaching', '')}"),
        kind="info",
        citations=[g["url"]] if g.get("url") else [],
        turn=turn,
    )


def submit_diagnosis(state: GameState, disease: dict, diagnosis: str,
                     differentials: Optional[List[str]] = None,
                     confidence: Optional[int] = None) -> List[Message]:
    if not normalize(diagnosis):  # blank guess: don't score or burn a retry
        return [Message(role="critic", text="Please enter a diagnosis.",
                        kind="info", turn=state.turn)]
    differentials = [d for d in (differentials or []) if normalize(d)]
    state.diagnosis = diagnosis
    state.diagnosis_differentials = differentials
    state.diagnosis_confidence = confidence
    citation = disease.get("guideline", {}).get("url")
    mult = safety.severity_multiplier(disease)   # scale diagnosis stakes by case severity
    messages: List[Message] = []
    correct = _is_correct(diagnosis, disease)

    # --- Correct primary → win ------------------------------------------- #
    if correct:
        delta = apply_points(state, "correct_diagnosis", AGENT, scale=mult,
                             reason=f"Correctly diagnosed {disease['name']}.",
                             citation=citation)
        state.status = "complete"
        messages.append(Message(role="critic",
                                text=f"Correct — this is {disease['name']}.",
                                kind="success", points_delta=delta,
                                citations=[citation] if citation else [],
                                turn=state.turn))
        messages += _calibration_events(state, True, confidence)
        messages.append(_reveal(disease, state.turn))
        return messages

    # --- Not correct → consumes a retry ---------------------------------- #
    state.retries += 1
    remaining = state.max_retries - state.retries
    rank = _true_dx_rank(differentials, disease)

    if rank:  # wrong primary, but the true dx was in their ranked differentials
        credit = _RANK_CREDIT.get(rank, 6)
        delta = apply_points(state, "differential_recognition", AGENT, delta=credit,
                             reason=f"Listed the correct diagnosis as differential #{rank}.",
                             citation=citation)
        text = (f"'{diagnosis}' isn't the primary diagnosis — but you correctly ranked it "
                f"as differential #{rank}. Recognition credit awarded.")
        kind = "info"
    elif _is_partial(diagnosis, disease):
        delta = apply_points(state, "partial_diagnosis", AGENT,
                             reason=f"'{diagnosis}' is a reasonable differential but not the diagnosis.",
                             citation=citation)
        text = (f"Close — '{diagnosis}' is a plausible differential, but not the "
                f"primary diagnosis. Partial credit awarded.")
        kind = "info"
    else:
        delta = apply_points(state, "incorrect_diagnosis", AGENT, scale=mult,
                             reason=f"'{diagnosis}' is incorrect.", citation=citation)
        text = f"'{diagnosis}' is not correct."
        kind = "loss"

    # Premature closure: committed a flat-wrong dx with no objective work-up.
    flat_wrong = not rank and not _is_partial(diagnosis, disease)
    premature = (flat_wrong and disease.get("appropriate_tests")
                 and not state.ordered_tests and not state.performed_exam)
    prem_msg = None
    if premature:
        pd = apply_points(state, "premature_closure", AGENT,
                          reason="Committed to a diagnosis with no tests ordered or exams performed.")
        prem_msg = Message(role="critic", kind="loss", points_delta=pd, turn=state.turn,
                           text="Premature closure — you committed before gathering any objective data "
                                "(no tests or exams).")

    cal_msgs = _calibration_events(state, False, confidence)

    if remaining <= 0:
        state.status = "failed"
        text += " No diagnosis attempts remain — the case is closed."
        messages.append(Message(role="critic", text=text, kind="fail",
                                points_delta=delta, turn=state.turn))
    else:
        text += f" You have {remaining} diagnosis attempt(s) left — keep working it up."
        messages.append(Message(role="critic", text=text, kind=kind,
                                points_delta=delta, turn=state.turn))
    if prem_msg:
        messages.append(prem_msg)
    messages += cal_msgs
    if remaining <= 0:
        messages.append(_reveal(disease, state.turn))
    return messages

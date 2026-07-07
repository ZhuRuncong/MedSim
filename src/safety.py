"""Severity weighting + executable safety gates.

Two clinician-credibility mechanisms, both deterministic and auditable:

- **Severity weighting** — a case's ``severity`` (1 routine → 3 life-threatening,
  authored by the generator) scales the stakes of the terminal decision and of
  unsafe actions. Missing an inferior STEMI should not cost the same as
  mislabelling a benign rash. Absent ``severity`` ⇒ multiplier 1.0 (unchanged).

- **Executable gates** — a case's ``safety_gates`` encode *ordered* clinical
  dependencies the student must respect, e.g. "check potassium (BMP) before
  starting insulin" or "confirm pregnancy (β-hCG) before imaging in a
  child-bearing patient". Doing the risky action without the prerequisite fires
  a one-time, severity-scaled penalty. Gate types:
    * ``test_before_drug`` — {test, drug}: prescribing ``drug`` needs ``test`` ordered first.
    * ``test_before_test`` — {first, then}: ordering ``then`` needs ``first`` ordered first.
"""
from __future__ import annotations

from typing import List

from .scoring import apply_points
from .state import GameState, Message

_SEV_MULT = {1: 1.0, 2: 1.5, 3: 2.0}

GATE_TYPES = ("test_before_drug", "test_before_test")


def severity_multiplier(disease: dict) -> float:
    """Scoring multiplier for a case's clinical stakes (1.0 if unspecified)."""
    s = disease.get("severity")
    if s is None:
        return 1.0
    try:
        return _SEV_MULT.get(int(s), 1.0)
    except (TypeError, ValueError):
        return 1.0


def _violated(gate: dict, trigger_type: str, value: str, state: GameState) -> bool:
    gtype = gate.get("type")
    if gtype == "test_before_drug" and trigger_type == "prescribe_drug" and value == gate.get("drug"):
        return gate.get("test") not in state.ordered_tests
    if gtype == "test_before_test" and trigger_type == "order_test" and value == gate.get("then"):
        return gate.get("first") not in state.ordered_tests
    return False


def check_gates(state: GameState, disease: dict, trigger_type: str, value: str) -> List[Message]:
    """Evaluate safety gates triggered by the action just taken; apply a
    severity-scaled penalty (once per gate) and return violation messages."""
    msgs: List[Message] = []
    mult = severity_multiplier(disease)
    citation = disease.get("guideline", {}).get("url")
    for i, gate in enumerate(disease.get("safety_gates", []) or []):
        gid = gate.get("id") or f"gate:{i}"
        if gid in state.gates_fired:
            continue
        if _violated(gate, trigger_type, value, state):
            state.gates_fired.append(gid)
            base = int(gate.get("penalty", 10) or 10)
            delta = apply_points(state, "safety_gate_violation", "SafetyAgent",
                                 delta=-round(base * mult),
                                 reason=gate.get("rationale", "Safety sequence violated."),
                                 citation=citation)
            msgs.append(Message(role="safety", kind="fail", points_delta=delta, turn=state.turn,
                                text="Safety gate — " + gate.get("rationale", "ordered dependency violated."),
                                citations=[citation] if citation else []))
    return msgs

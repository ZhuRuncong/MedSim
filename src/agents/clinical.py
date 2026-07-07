"""ClinicalAgent (PLAN §5): lab ordering & drug prescription.

Calls the lab_simulator and drug_effect_engine tools, records results/drugs in
the state, and awards points per the evidence-based rubric.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .. import data_loader, safety
from ..scoring import apply_points
from ..state import GameState, Message
from ..tools import evaluate_drug, simulate_labs

AGENT = "ClinicalAgent"


def _navigated_allergy(disease: dict, allergies) -> bool:
    """True if a guideline first-line option WOULD have conflicted with one of
    the patient's allergies — i.e. choosing a safe drug reflects a real tradeoff
    rather than an incidental, unrelated allergy."""
    if not allergies:
        return False
    return any(data_loader.allergy_conflict(dr, allergies)
               for dr in disease.get("first_line_drugs", []))


def _format_result(test: str, result: dict) -> str:
    kind = result.get("type")
    if kind == "panel":
        abnormal = []
        for comp, r in result["components"].items():
            flag = r["flag"]
            if flag != "N":
                abnormal.append(f"{comp} {r['value']} {r['unit']} [{flag}] "
                                f"(ref {r['ref_low']}-{r['ref_high']})")
        if abnormal:
            return f"{test}: " + "; ".join(abnormal)
        return f"{test}: all components within normal limits."
    if kind == "qualitative":
        return f"{test}: {result.get('finding')}"
    if kind == "unavailable":
        return result.get("note", f"{test} unavailable.")
    return f"{test}: {result}"


def order_test(state: GameState, disease: dict, test: str,
               rng: np.random.Generator) -> List[Message]:
    if test in state.ordered_tests:
        apply_points(state, "duplicate_action", AGENT,
                     reason=f"{test} was already ordered.")
        return [Message(role="clinical",
                        text=f"{test} was already ordered — see earlier results.",
                        kind="loss", points_delta=0, turn=state.turn)]

    out = simulate_labs([test], disease, state.vitals, rng)
    result = out["results"][test]
    state.ordered_tests.append(test)
    state.test_results[test] = result

    if result.get("type") == "unavailable":
        return [Message(role="clinical", text=_format_result(test, result),
                        kind="info", turn=state.turn)]

    appropriate = test in disease.get("appropriate_tests", [])
    action = "order_appropriate_test" if appropriate else "order_unnecessary_test"
    citation = disease.get("guideline", {}).get("url")
    reason = ("Indicated in the diagnostic work-up." if appropriate
              else "Low-value / not indicated for this presentation.")
    delta = apply_points(state, action, AGENT, reason=reason, citation=citation)

    text = _format_result(test, result)
    msgs = [Message(role="clinical", text=text,
                    kind="gain" if delta > 0 else "loss",
                    points_delta=delta,
                    citations=out.get("citations", []),
                    turn=state.turn)]
    msgs += safety.check_gates(state, disease, "order_test", test)
    return msgs


def prescribe_drug(state: GameState, disease: dict, drug: str) -> List[Message]:
    if drug in state.prescribed_drugs:
        apply_points(state, "duplicate_action", AGENT,
                     reason=f"{drug} was already prescribed.")
        return [Message(role="clinical",
                        text=f"{drug} has already been prescribed.",
                        kind="loss", turn=state.turn)]

    ev = evaluate_drug(drug, disease, state.allergies, state.prescribed_drugs)
    state.prescribed_drugs.append(drug)
    citation = disease.get("guideline", {}).get("url")
    mult = safety.severity_multiplier(disease)   # harm is worse in a higher-stakes case
    category = ev["category"]
    allergy = ev.get("allergy")
    messages: List[Message] = []

    if category == "allergen":
        delta = apply_points(state, "prescribe_allergen", AGENT, scale=mult,
                             reason=ev["rationale"], citation=citation)
        kind = "fail"
    elif category == "contraindicated":
        delta = apply_points(state, "prescribe_contraindicated_drug", AGENT, scale=mult,
                             reason=ev["rationale"], citation=citation)
        kind = "fail"
    elif category == "first_line":
        delta = apply_points(state, "prescribe_guideline_first_line", AGENT,
                             reason=ev["rationale"], citation=citation)
        kind = "gain"
        # Reward avoiding an allergy only when a first-line option genuinely
        # conflicted AND this drug is fully clear (no allergen/cross-reactivity).
        if allergy is None and _navigated_allergy(disease, state.allergies):
            delta += apply_points(state, "avoid_allergy", AGENT,
                                  reason="Chose an effective first-line agent that avoids the patient's allergy.")
    elif category == "reasonable":
        delta = apply_points(state, "prescribe_reasonable_drug", AGENT,
                             reason=ev["rationale"], citation=citation)
        kind = "gain"
    else:  # neutral
        delta = 0
        kind = "info"

    text = f"Prescribed {drug}. {ev['rationale']}"
    messages.append(Message(role="clinical", text=text, kind=kind,
                            points_delta=delta, citations=ev.get("citations", []),
                            turn=state.turn))

    # Surface a cross-reactivity caution instead of silently dropping it.
    if allergy and allergy.get("type") == "cross_reactivity":
        messages.append(Message(
            role="clinical",
            text=f"Allergy caution: {allergy.get('description', '')}",
            kind="loss", citations=[citation] if citation else [], turn=state.turn))

    # Surface drug-interaction warnings; major ones carry an extra safety penalty.
    for it in ev.get("interactions", []):
        sev = it.get("severity")
        warn = f"{sev.title()} interaction with {it['with']}: {it['description']}"
        extra = 0
        wkind = "info"
        if sev == "major":
            extra = apply_points(state, "major_drug_interaction", AGENT,
                                 reason=warn, citation=citation, delta=-5)
            wkind = "loss"
        messages.append(Message(role="clinical", text=warn, kind=wkind,
                                points_delta=extra, turn=state.turn))

    messages += safety.check_gates(state, disease, "prescribe_drug", drug)
    return messages

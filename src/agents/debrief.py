"""DebriefAgent — end-of-case after-action report.

When a case ends (complete/failed), this diffs the student's actual trajectory
against the case's answer key (which already lives in the disease record and the
score_log) to produce a structured :class:`Debrief`: what they hit, missed, and
did that was low-value or harmful; an efficiency score vs. the ideal work-up; the
key discriminating findings; and targeted, guideline-cited teaching points.

Everything is deterministic. When an LLM is configured it additionally voices a
short attending-style wrap-up, grounded ONLY in the computed diff (same pattern
as the Patient/Knowledge agents) — it can't invent clinical facts.
"""
from __future__ import annotations

import json
from typing import List, Optional

from .. import config, data_loader
from ..state import Debrief, GameState


def _ideal_points(disease: dict) -> int:
    p = config.POINTS
    mx = p["correct_diagnosis"]
    mx += p["order_appropriate_test"] * len(disease.get("appropriate_tests", []))
    mx += p["prescribe_guideline_first_line"]                     # one first-line agent
    mx += p["informative_exam"] * len(disease.get("appropriate_exams", []))
    if disease.get("indicated_surgeries"):
        mx += p["emergency_surgery_indicated"]
    return max(1, mx)


def _key_findings(disease: dict) -> List[str]:
    out = []
    for k, v in disease.get("lab_deviations", {}).items():
        if isinstance(v, dict) and v.get("finding"):
            out.append(f"{k}: {v['finding']}")
        elif isinstance(v, dict) and "value" in v:
            out.append(f"{k} = {v['value']} (abnormal)")
        if len(out) >= 3:
            break
    return out


def build_debrief(state: GameState, disease: dict, client=None) -> Debrief:
    appropriate_tests = disease.get("appropriate_tests", [])
    appropriate_exams = disease.get("appropriate_exams", [])
    first_line = disease.get("first_line_drugs", [])
    contraindicated = disease.get("contraindicated_drugs", [])
    indicated_surg = disease.get("indicated_surgeries", [])

    ordered, performed = state.ordered_tests, state.performed_exam
    prescribed, surgeries = state.prescribed_drugs, state.surgeries

    tests_hit = [t for t in appropriate_tests if t in ordered]
    tests_missed = [t for t in appropriate_tests if t not in ordered]
    tests_low_value = [t for t in ordered if t not in appropriate_tests]
    exams_hit = [e for e in appropriate_exams if e in performed]
    exams_missed = [e for e in appropriate_exams if e not in performed]

    fl_given = [d for d in prescribed if d in first_line]
    harmful = [d for d in prescribed if d in contraindicated]
    for d in prescribed:
        if d not in harmful and data_loader.allergy_conflict(d, state.allergies):
            harmful.append(d)
    missed_fl = [] if fl_given else first_line

    surgery_note = None
    if indicated_surg:
        got = [s for s in surgeries if s in indicated_surg]
        wrong = [s for s in surgeries if s not in indicated_surg]
        if got:
            surgery_note = f"Correctly requested {', '.join(got)}."
        elif wrong:
            surgery_note = f"Requested {', '.join(wrong)}, which was not indicated."
        else:
            surgery_note = f"This case needed {', '.join(indicated_surg)}; no operation was requested."
    elif surgeries:
        surgery_note = f"Requested {', '.join(surgeries)} — this condition is managed medically."

    # Safety flags
    safety = [f"Prescribed {d} — contraindicated or an allergen." for d in harmful]
    for s in surgeries:
        if indicated_surg and s not in indicated_surg or not indicated_surg:
            safety.append(f"Requested unindicated surgery: {s}.")
    if any(e.action == "premature_closure" for e in state.score_log):
        safety.append("Committed to a diagnosis before gathering objective data.")
    for e in state.score_log:
        if e.action == "safety_gate_violation":
            safety.append(f"Safety sequence violated: {e.reason}")

    # Teaching points (guideline-cited)
    teaching: List[str] = []
    if disease.get("teaching"):
        teaching.append(disease["teaching"])
    if missed_fl:
        teaching.append(f"Guideline first-line therapy: {', '.join(first_line[:4])}.")
    if tests_missed:
        teaching.append(f"Work-up you skipped: {', '.join(tests_missed[:4])}.")
    if harmful:
        teaching.append(f"Avoid {', '.join(harmful)} in this patient.")
    teaching = teaching[:4]

    outcome = "correct" if state.status == "complete" else "missed"
    mx = _ideal_points(disease)
    efficiency = round(min(1.0, max(0.0, state.points / mx)), 2)
    citation = disease.get("guideline", {}).get("url")

    d = Debrief(
        outcome=outcome, true_diagnosis=disease["name"], attempts=state.retries + (1 if outcome == "correct" else 0),
        turns=state.turn, points=state.points, max_points=mx, efficiency=efficiency,
        tests_hit=tests_hit, tests_missed=tests_missed, tests_low_value=tests_low_value,
        exams_hit=exams_hit, exams_missed=exams_missed,
        drugs_first_line=fl_given, drugs_missed_first_line=missed_fl, drugs_harmful=harmful,
        surgery_note=surgery_note, key_findings=_key_findings(disease),
        teaching_points=teaching, safety_flags=safety, citation=citation,
    )
    d.attending_note = _deterministic_note(d, disease)

    # Optional grounded LLM voice
    if client is None and config.debrief_llm_enabled():
        from ..llm import get_client
        client = get_client()
    if client is not None:
        try:
            note = _llm_note(client, d, disease)
            if note:
                d.attending_note = note
                d.llm = True
        except Exception:
            pass  # keep the deterministic note
    return d


def _deterministic_note(d: Debrief, disease: dict) -> str:
    lead = (f"Correct — this was {d.true_diagnosis}." if d.outcome == "correct"
            else f"The diagnosis was {d.true_diagnosis}.")
    eff = f" You reached {int(d.efficiency * 100)}% of the ideal work-up."
    tip = f" {disease.get('teaching', '')}".rstrip()
    if d.safety_flags:
        tip = f" Key issue: {d.safety_flags[0]}"
    return (lead + eff + tip).strip()


_SYSTEM = (
    "You are an attending physician giving a brief end-of-case debrief to a medical student. "
    "Be honest but encouraging. Use ONLY the facts in the provided report (outcome + the diff of "
    "what the student did vs. the ideal work-up). Do NOT invent clinical facts, drugs, or tests. "
    "3-4 sentences. Name the diagnosis (the case is over). "
    'Respond as JSON: {"note": "<your debrief>"}.'
)


def _llm_note(client, d: Debrief, disease: dict) -> str:
    report = {
        "diagnosis": d.true_diagnosis, "outcome": d.outcome,
        "efficiency_pct": int(d.efficiency * 100),
        "tests_ordered_correctly": d.tests_hit, "tests_missed": d.tests_missed,
        "low_value_tests": d.tests_low_value,
        "first_line_given": d.drugs_first_line, "first_line_missed": d.drugs_missed_first_line,
        "harmful_drugs": d.drugs_harmful, "safety_flags": d.safety_flags,
        "teaching": disease.get("teaching", ""),
    }
    out = client.complete_json(_SYSTEM, "Debrief this encounter:\n" + json.dumps(report, indent=2),
                               temperature=0.4)
    return (out or {}).get("note", "").strip()


def attach(state: GameState, disease: dict, client=None) -> None:
    """Compute & attach the debrief to a finished case (idempotent)."""
    if state.status in ("complete", "failed") and state.debrief is None:
        state.debrief = build_debrief(state, disease, client=client)

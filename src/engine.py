"""High-level game engine — the façade the UI, API and tests call.

Wraps case creation, action dispatch (via the SupervisorAgent) and session
persistence. Deterministic given a case id, so integration tests are stable.
"""
from __future__ import annotations

import os
import uuid
from typing import List, Optional

import numpy as np

from . import data_loader, trace as trace_mod
from .agents import patient, supervisor
from .config import level_for_points
from .state import GameState, Message
from .store import SessionStore
from .util import case_rng, redact_citation, redact_diagnosis


# --------------------------------------------------------------------------- #
# Option lists for the UI
# --------------------------------------------------------------------------- #
def available_tests() -> List[str]:
    return data_loader.orderable_tests()


def available_drugs() -> List[str]:
    return data_loader.all_drug_names()


_DISTRACTOR_PROCEDURES = [
    "Exploratory Laparotomy", "Craniotomy", "Thoracotomy", "Amputation",
    "Splenectomy", "Tracheostomy",
]


def available_procedures() -> List[str]:
    """All indicated procedures across the catalog + distractors.

    Deliberately not filtered to the current disease so the choice is a real
    clinical decision rather than a give-away.
    """
    procs = set(_DISTRACTOR_PROCEDURES)
    for d in data_loader.diseases():
        procs.update(d.get("indicated_surgeries", []))
    return sorted(procs)


# --------------------------------------------------------------------------- #
# Case lifecycle
# --------------------------------------------------------------------------- #
def create_case(
    specialties: List[str],
    student_id: str = "guest",
    student_name: str = "Student",
    level: int = 1,
    store: Optional[SessionStore] = None,
    disease_id: Optional[str] = None,
) -> GameState:
    """Build a case for a registered disease.

    Cases are AI-generated (``create_generated_case``); pass ``disease_id`` to
    build state for a specific registered disease. Without an id, a random
    already-registered disease for the specialties is used (e.g. replaying a
    previously-generated case).
    """
    if disease_id is not None:
        disease = data_loader.get_disease(disease_id)
        if disease is None:  # self-heal from the on-disk generated cache (see perform_action)
            data_loader.ensure_generated_loaded(force=True)
            disease = data_loader.get_disease(disease_id)
        if disease is None:
            raise ValueError(f"Unknown disease_id '{disease_id}'.")
    else:
        pool = [d for d in data_loader.diseases_for_specialties(specialties) if d]
        if not pool:
            raise ValueError(
                "No cases available — MedSim generates every case with AI. "
                "Use create_generated_case() (an API key is required).")
        picker = np.random.default_rng()
        disease = pool[int(picker.integers(0, len(pool)))]

    # Report the case under one of the specialties the student actually selected.
    match = [s for s in disease.get("specialties", []) if s in set(specialties)]
    specialty = match[0] if match else disease.get("specialties", ["General"])[0]

    case_id = uuid.uuid4().hex[:8]
    state = GameState(
        case_id=case_id,
        disease_id=disease["id"],
        disease_name=disease["name"],
        specialty=specialty,
        student_id=student_id,
        student_name=student_name,
        level=level,
    )

    rng = case_rng(case_id)
    intro = patient.intake(state, disease, rng)
    # Never let the intake copy or chief complaint name the hidden diagnosis.
    state.chief_complaint = redact_diagnosis(state.chief_complaint, disease)
    for m in intro:
        m.text = redact_diagnosis(m.text, disease)
    state.feed.extend(intro)

    if store is not None:
        store.save(state)
    return state


def create_generated_case(
    specialty: str,
    difficulty: Optional[int] = None,
    student_id: str = "guest",
    student_name: str = "Student",
    level: int = 1,
    store: Optional[SessionStore] = None,
    client=None,
) -> GameState:
    """Generate a brand-new, LLM-authored + verified case and start it.

    Raises llm.LLMUnavailable / generator.CaseGenerationError on failure — the
    caller (UI) surfaces these to the student.
    """
    from . import generator  # local import: keeps optional LLM deps lazy

    disease = generator.generate_case(specialty, difficulty, client=client)
    return create_case(
        [specialty], student_id=student_id, student_name=student_name,
        level=level, store=store, disease_id=disease["id"],
    )


_compiled_graph = None


def _use_langgraph() -> bool:
    """Route actions through the compiled LangGraph when it is installed.

    Opt out with MEDSIM_USE_LANGGRAPH=0. Both paths share the same handler
    functions, so behaviour is identical either way (see graph.py).
    """
    if os.getenv("MEDSIM_USE_LANGGRAPH", "1") == "0":
        return False
    from .agents import graph
    return graph.LANGGRAPH_AVAILABLE


def _route(state: GameState, disease: dict, action: dict, rng) -> List[Message]:
    global _compiled_graph
    if _use_langgraph():
        from .agents import graph
        if _compiled_graph is None:
            _compiled_graph = graph.build_graph()
        out = _compiled_graph.invoke({
            "game_state": state, "disease": disease, "action": action, "rng": rng,
        })
        return out.get("messages", [])
    return supervisor.route(state, disease, action, rng)


def perform_action(
    state: GameState,
    action: dict,
    store: Optional[SessionStore] = None,
) -> List[Message]:
    """Run one student action through the supervisor and return new messages.

    Every action is recorded as a TraceEvent (agent trajectory) on the state.
    """
    disease = data_loader.get_disease(state.disease_id)
    if disease is None:
        # Self-heal: the in-memory registry can be cleared (Streamlit hot-reload,
        # a fresh process) while a persisted session still points at a generated
        # case. Reload the on-disk generated cache and retry before giving up.
        data_loader.ensure_generated_loaded(force=True)
        disease = data_loader.get_disease(state.disease_id)
    if disease is None:
        raise ValueError(f"Case references unknown disease '{state.disease_id}'.")

    rng = case_rng(state.case_id, "action", state.turn)
    with trace_mod.record(state, action) as ev:
        messages = _route(state, disease, action, rng)
        ev.llm_used = any(m.llm for m in messages)

    # Leak guard: while the case is still open, no feedback may name the hidden
    # diagnosis (drug/surgery rationales, lab findings, knowledge answers, …).
    # Once the case closes (correct dx or retries exhausted) the reveal is allowed.
    if state.status == "active":
        for m in messages:
            m.text = redact_diagnosis(m.text, disease)
            m.citations = [redact_citation(c, disease) for c in m.citations]

    # End-of-case debrief (fires once, the moment the case closes).
    if state.status in ("complete", "failed") and state.debrief is None:
        from .agents import debrief as _debrief
        _debrief.attach(state, disease)
        messages.append(Message(role="critic", kind="info", turn=state.turn,
                                text="Case debrief ready — see the after-action report below."))

    state.feed.extend(messages)
    state.level = max(state.level, level_for_points(state.points))
    if store is not None:
        store.save(state)
    return messages


# --------------------------------------------------------------------------- #
# Workflow diagram (PLAN §5)
# --------------------------------------------------------------------------- #
MERMAID = """graph TD
    START[Create New Case] --> PAT[PatientAgent]
    PAT --> SUP[SupervisorAgent]
    SUP -->|Ask History| PAT
    SUP -->|Perform Exam| PAT
    SUP -->|Order Test| CLIN[ClinicalAgent]
    SUP -->|Prescribe Drug| CLIN
    SUP -->|Request Surgery| SURG[SurgeryAgent]
    SUP -->|Ask Knowledge Q| KNOW[KnowledgeAgent]
    SUP -->|Submit Diagnosis| CRIT[CriticAgent]
    CRIT -->|Correct| DONE[Case Complete - Level Up]
    CRIT -->|Incorrect| SUP
    SUP -->|Max Retries| DONE
"""

"""Static configuration: specialties, the evidence-based scoring rubric,
file paths, and runtime settings.

This module has *no* heavy imports so it is safe to import from anywhere
(agents, tools, UI, tests) without triggering optional dependencies.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

DRUG_INTERACTIONS_JSON = DATA_DIR / "drug_interactions.json"
LAB_REFS_CSV = DATA_DIR / "lab_refs.csv"
GENERATED_DIR = DATA_DIR / "generated"   # AI-generated, verified cases are cached here

# --------------------------------------------------------------------------- #
# Specialties (drive the UI checkbox list & case selection)
# --------------------------------------------------------------------------- #
SPECIALTIES = [
    "Internal Medicine",
    "Emergency",
    "Pediatrics",
    "Dermatology",
    "OB-GYN",
    "Surgery",
]

# --------------------------------------------------------------------------- #
# Evidence-based scoring rubric (§7 of PLAN.md).
# Positive = guideline-concordant action; negative = harm / low-value care.
# --------------------------------------------------------------------------- #
POINTS = {
    "order_appropriate_test": 10,
    "order_unnecessary_test": -5,
    "duplicate_action": -2,
    "prescribe_guideline_first_line": 15,
    "prescribe_reasonable_drug": 3,
    "prescribe_contraindicated_drug": -20,
    "prescribe_allergen": -20,
    "avoid_allergy": 5,
    "correct_diagnosis": 30,
    "partial_diagnosis": 10,   # named a plausible differential
    "incorrect_diagnosis": -15,
    # --- metacognition (ranked differentials + confidence calibration) ---
    "differential_recognition": 8,   # listed the true dx among ranked differentials (scaled by rank)
    "calibrated_confidence": 5,      # correct AND appropriately confident
    "overconfident_error": -8,       # wrong AND highly confident (anchoring)
    "prudent_uncertainty": 2,        # wrong BUT appropriately unsure (rewarded hedging)
    "premature_closure": -5,         # committed with no tests/exams performed
    "emergency_surgery_indicated": 20,
    "unwarranted_surgery": -25,
    "safety_gate_violation": -10,   # violated an ordered safety dependency (base; per-gate + severity scaled)
    "use_knowledge_query": 2,
    "informative_history": 1,  # asked a question the patient could answer
    "informative_exam": 1,     # performed an exam that revealed a finding
}

# --------------------------------------------------------------------------- #
# Flow-control settings
# --------------------------------------------------------------------------- #
MAX_RETRIES = 3          # incorrect-diagnosis attempts before the case fails
POINTS_PER_LEVEL = 100   # student levels up every N cumulative points

# Canonical physical-exam actions offered in the UI. Disease `exam_findings`
# keys are drawn from this list so the two always agree.
PHYSICAL_EXAMS = [
    "General inspection",
    "Inspect skin",
    "Auscultate heart",
    "Auscultate lungs",
    "Percuss chest",
    "Palpate abdomen",
    "Palpate lymph nodes",
    "Palpate thyroid",
    "Examine throat",
    "Inspect oral cavity",
    "Check extremities/edema",
    "Palpate peripheral pulses",
    "Assess JVP",
    "Assess capillary refill",
    "Assess hydration status",
    "Neurological exam",
    "Test deep tendon reflexes",
    "Cranial nerve exam",
    "Assess gait and coordination",
    "Assess mental status",
    "Examine joints / range of motion",
    "Fundoscopy",
    "Otoscopy",
    "Pelvic exam",
    "Digital rectal exam",
    "Breast exam",
]

# --------------------------------------------------------------------------- #
# Runtime / integration settings (all optional, read from environment)
# --------------------------------------------------------------------------- #
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "")
MEDSIM_API_URL = os.getenv("MEDSIM_API_URL", "http://localhost:8000")

# --- LLM providers for the AI case generator (all optional) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
# "gemini" | "anthropic" | "" (auto-select by whichever key is present)
LLM_PROVIDER = os.getenv("MEDSIM_LLM_PROVIDER", "")

# Case-generation safety knobs (generate-then-verify harness)
GEN_MAX_ATTEMPTS = int(os.getenv("MEDSIM_GEN_MAX_ATTEMPTS", "4"))
GEN_VERIFIERS = int(os.getenv("MEDSIM_GEN_VERIFIERS", "2"))       # independent reviewers per case
GEN_MIN_CONFIDENCE = float(os.getenv("MEDSIM_GEN_MIN_CONFIDENCE", "0.7"))
# Verify a generated case's guideline URL actually resolves (kills fabricated DOIs).
CHECK_CITATIONS = os.getenv("MEDSIM_CHECK_CITATIONS", "1") != "0"

# Feature flags (auto-detected; can be forced off via env)
USE_LLM = bool(GOOGLE_API_KEY) and os.getenv("MEDSIM_DISABLE_LLM") != "1"
USE_TAVILY = bool(TAVILY_API_KEY)
USE_REDIS = bool(REDIS_URL)


def llm_available() -> bool:
    """True if any LLM provider is configured (enables AI case generation)."""
    return bool(GOOGLE_API_KEY or ANTHROPIC_API_KEY)


def patient_llm_enabled() -> bool:
    """True if the PatientAgent should phrase replies with an LLM.

    Grounded generation: the LLM only restyles facts the deterministic engine
    already chose, so it never invents findings or leaks the diagnosis.
    """
    return llm_available() and os.getenv("MEDSIM_DISABLE_PATIENT_LLM") != "1"


def knowledge_llm_enabled() -> bool:
    """True if the KnowledgeAgent should synthesize answers with an LLM.

    Retrieval-augmented: the LLM answers ONLY from the source the deterministic
    lookup retrieved, so it can't fabricate facts or citations.
    """
    return llm_available() and os.getenv("MEDSIM_DISABLE_KNOWLEDGE_LLM") != "1"


def debrief_llm_enabled() -> bool:
    """True if the end-of-case debrief should be voiced by an LLM.

    Grounded: the LLM only rephrases the deterministically-computed diff of the
    student's work-up vs. the ideal path.
    """
    return llm_available() and os.getenv("MEDSIM_DISABLE_DEBRIEF_LLM") != "1"


def level_for_points(points: int) -> int:
    """Return the 1-indexed student level for a cumulative point total."""
    if points <= 0:
        return 1
    return 1 + points // POINTS_PER_LEVEL

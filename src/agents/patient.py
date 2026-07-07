"""PatientAgent (PLAN §5).

Generates the intake (age, sex, chief complaint, vitals, allergies) and answers
history questions & physical-exam requests from the hidden case script.
Deterministic keyword matching by default; an LLM can enrich phrasing when a key
is configured (kept optional so the app runs offline).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import config
from ..state import GameState, Message
from ..tools import generate_symptoms
from ..util import words

# Allergy prevalence for generated patients (makes the drug engine meaningful).
_ALLERGY_CHOICES = ["none", "penicillin", "sulfa", "nsaid", "opioid"]
_ALLERGY_P = [0.60, 0.18, 0.10, 0.07, 0.05]

_STOP = {"do", "you", "have", "any", "is", "are", "the", "a", "an", "of", "to",
         "and", "with", "your", "did", "does", "has", "had", "been", "was",
         "were", "feel", "feeling", "there", "i", "me", "my", "for", "on", "in"}

# Maps a history topic -> keywords that should trigger that answer.
_HISTORY_KEYWORDS = {
    "onset": ["onset", "start", "started", "begin", "began", "how long", "since", "when"],
    "duration": ["how long", "duration"],
    "cough": ["cough", "sputum", "phlegm"],
    "fever": ["fever", "temperature", "chills", "rigors", "hot"],
    "chest pain": ["chest"],
    "radiation": ["radiate", "radiates", "spread", "arm", "jaw"],
    "breathing": ["breath", "breathing", "dyspnea", "wheeze", "wheezing", "winded", "short of breath"],
    "abdominal pain": ["abdomen", "abdominal", "belly", "stomach", "tummy"],
    "appetite": ["appetite", "eating", "nausea", "vomit", "vomiting"],
    "bowel": ["bowel", "stool", "diarrhea", "constipation"],
    "urination": ["urine", "urinate", "urination", "pee", "dysuria", "peeing"],
    "menstrual": ["period", "menstrual", "menstruation", "lmp"],
    "pregnancy": ["pregnant", "pregnancy"],
    "smoking": ["smoke", "smoking", "tobacco", "cigarette", "cigarettes"],
    "alcohol": ["alcohol", "drink", "drinking"],
    "medications": ["medication", "medications", "medicine", "meds", "pills", "taking"],
    "family history": ["family", "father", "mother", "hereditary", "genetic"],
    "travel": ["travel", "trip", "flight"],
    "sick contacts": ["sick", "contact", "contacts", "exposure", "exposed"],
    "triggers": ["trigger", "triggers", "worse", "exacerbate", "brings on", "cause"],
    "vision": ["vision", "sight", "blur", "blurry", "eyes", "spots"],
    "swelling": ["swell", "swelling", "edema", "puffy", "puffed"],
    "skin": ["rash", "skin", "red", "redness"],
    "itch": ["itch", "itching", "itchy", "scratch"],
    "injury": ["injury", "injured", "cut", "scrape", "wound", "trauma"],
    "diabetes": ["diabetes", "diabetic", "sugar"],
    "thirst": ["thirst", "thirsty", "dry mouth"],
    "exertion": ["exertion", "exercise", "activity", "walking", "exert"],
    "sexual history": ["sexual", "partner", "partners", "std", "sti", "chlamydia"],
    "prenatal care": ["prenatal", "antenatal", "checkup"],
    "prior episodes": ["before", "previous", "prior", "recurrent", "again"],
    "jaundice": ["jaundice", "yellow", "dark urine"],
    "feeding": ["feeding", "feed", "bottle"],
    "wet diapers": ["diaper", "diapers", "wet"],
    "dizziness": ["dizzy", "dizziness", "faint", "lightheaded"],
}


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #
def intake(state: GameState, disease: dict, rng: np.random.Generator) -> List[Message]:
    demo = disease.get("demographics", {})
    lo, hi = demo.get("age_range", [30, 60])
    if hi <= 2:
        age_str = f"{int(rng.integers(1, 19))}-month-old"
    else:
        age_str = f"{int(rng.integers(lo, hi + 1))}-year-old"

    sex_pref = demo.get("sex", "any")
    sex = sex_pref if sex_pref in ("male", "female") else rng.choice(["male", "female"])
    state.demographics = {"age": age_str, "sex": sex}

    allergy = rng.choice(_ALLERGY_CHOICES, p=_ALLERGY_P)
    state.allergies = [] if allergy == "none" else [str(allergy)]

    # Vitals with mild patient-to-patient jitter.
    base = disease.get("vitals", {})
    vitals = {}
    for k, v in base.items():
        if k == "Temp":
            vitals[k] = round(float(v) + float(rng.normal(0, 0.15)), 1)
        elif k == "SpO2":
            vitals[k] = min(100.0, round(float(v) + float(rng.integers(-1, 2)), 0))
        else:
            vitals[k] = round(float(v) + float(rng.integers(-2, 3)), 0)
    state.vitals = vitals

    sym = generate_symptoms(disease, rng)
    state.symptoms = sym["present_symptoms"]
    state.chief_complaint = disease.get("chief_complaint", "")

    intro = (f"A {age_str} {sex} presents to the {state.specialty} service. "
             f"Chief complaint: \"{state.chief_complaint}\".")
    allergy_line = ("Charted allergies: " + ", ".join(state.allergies) + "."
                    if state.allergies else "Charted allergies: none known.")
    return [
        Message(role="patient", text=intro, kind="info", turn=state.turn),
        Message(role="system", text=allergy_line, kind="info", turn=state.turn),
    ]


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def _match_topic(question: str, disease: dict) -> Optional[str]:
    q = question.lower()
    qtokens = words(q)
    history = disease.get("history", {})
    best_topic, best_len = None, 0
    for topic in history:
        for kw in _HISTORY_KEYWORDS.get(topic, [topic]):
            # Multi-word keywords match as a phrase; single words must match on a
            # whole-word boundary so 'arm' does not match inside 'warm'.
            hit = (kw in q) if " " in kw else (kw in qtokens)
            if hit and len(kw) > best_len:
                best_topic, best_len = topic, len(kw)
    return best_topic


def _symptom_reply(question: str, state: GameState, disease: dict) -> Optional[str]:
    qtokens = words(question) - _STOP
    if not qtokens:
        return None
    present = state.symptoms
    all_syms = [s["name"] for s in disease.get("symptoms", [])]
    for name in present:
        if qtokens & (words(name) - _STOP):
            return f"Yes — {name}."
    for name in all_syms:
        if name not in present and (qtokens & (words(name) - _STOP)):
            return "No, I haven't noticed anything like that."
    return None


def _deterministic_answer(state: GameState, disease: dict, question: str) -> dict:
    """Rule-based grounding: decides *what* is true and whether it's informative
    (this drives scoring and always runs, LLM or not)."""
    q = question.lower()
    if any(w in q for w in ("allerg",)):
        if state.allergies:
            return {"answer": f"Yes — I'm allergic to {', '.join(state.allergies)}.",
                    "informative": True}
        return {"answer": "No known drug allergies.", "informative": True}

    topic = _match_topic(question, disease)
    if topic:
        return {"answer": disease["history"][topic], "informative": True, "topic": topic}

    sym = _symptom_reply(question, state, disease)
    if sym:
        return {"answer": sym, "informative": True}

    return {"answer": "Hmm, I'm not sure — could you ask that a different way?",
            "informative": False}


def answer_history(state: GameState, disease: dict, question: str, client=None) -> dict:
    """Answer a history question. The rule-based layer grounds the facts &
    scoring; when an LLM is available it re-voices the reply naturally.

    ``client`` is injectable for tests; in production it is resolved from the
    configured provider only when patient-LLM is enabled.
    """
    base = _deterministic_answer(state, disease, question)

    if client is None and config.patient_llm_enabled():
        from ..llm import get_client
        client = get_client()

    if client is not None:
        try:
            reply = _llm_reply(client, state, disease, question, base)
            if reply:
                base = {**base, "answer": reply, "llm": True}
        except Exception:
            pass  # any failure → deterministic answer already in `base`
    return base


# --------------------------------------------------------------------------- #
# Grounded LLM phrasing (optional)
# --------------------------------------------------------------------------- #
_PATIENT_SYSTEM = (
    "You are role-playing a patient in a medical-education simulation. Speak in the "
    "first person, in plain lay language, in 1-3 short sentences, matching the emotional "
    "tone of someone with your presentation. Answer ONLY using facts in the provided "
    "chart. If asked about anything not in the chart, briefly say you don't have that "
    "problem or don't know — never invent symptoms, test results, or medical facts. "
    "NEVER state, guess, or hint at your diagnosis or name any disease. "
    'Respond as JSON: {"reply": "<what you say>"}.'
)


def _build_chart(state: GameState, disease: dict) -> str:
    present = state.symptoms or []
    all_syms = [s["name"] for s in disease.get("symptoms", [])]
    absent = [s for s in all_syms if s not in present]
    demo = state.demographics
    facts = "; ".join(disease.get("history", {}).values())
    return "\n".join([
        f"- You are a {demo.get('age', '')} {demo.get('sex', '')}.",
        f"- Reason for visit: {state.chief_complaint}",
        f"- Symptoms you HAVE: {', '.join(present) or 'none in particular'}.",
        f"- Symptoms you do NOT have: {', '.join(absent) or 'n/a'}; and anything not "
        f"listed here, you do not have.",
        f"- Known drug allergies: {', '.join(state.allergies) or 'none'}.",
        f"- Other true things about you: {facts or 'nothing further'}.",
    ])


def _llm_reply(client, state: GameState, disease: dict, question: str, base: dict) -> str:
    truthful = base.get("answer", "")
    guidance = (f"A truthful answer to convey is: \"{truthful}\". Say it naturally in your "
                f"own words." if base.get("informative")
                else "If this isn't something you'd know about or have, briefly say so.")
    user = (f"CHART (private — do not read out or reveal you have a chart):\n"
            f"{_build_chart(state, disease)}\n\nThe clinician asks: \"{question}\"\n{guidance}")
    out = client.complete_json(_PATIENT_SYSTEM, user, temperature=0.7)
    reply = (out or {}).get("reply", "").strip()
    if reply and _leaks_diagnosis(reply, disease):
        return ""  # drop it → caller keeps the deterministic answer
    return reply


def _leaks_diagnosis(reply: str, disease: dict) -> bool:
    """True if the reply names the disease or one of its aliases as whole words.

    Whole-word (token-subset) matching catches 'pneumonia' / 'community-acquired
    pneumonia' without over-blocking ordinary words like 'chest'."""
    rtokens = words(reply)
    for name in [disease.get("name", ""), *disease.get("aliases", [])]:
        ntokens = words(name)
        if ntokens and ntokens <= rtokens:
            return True
    return False


# --------------------------------------------------------------------------- #
# Physical exam
# --------------------------------------------------------------------------- #
def perform_exam(state: GameState, disease: dict, exam: str) -> dict:
    findings = disease.get("exam_findings", {})
    if exam in findings:
        return {"finding": findings[exam], "informative": True}
    return {"finding": f"{exam}: no significant abnormality detected.",
            "informative": False}

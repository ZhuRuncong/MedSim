"""Severity weighting + executable safety gates."""
import numpy as np

from src import safety
from src.agents import clinical, critic, debrief
from src.state import GameState

RNG = np.random.default_rng(0)


def _state():
    return GameState(case_id="t", disease_id="X", disease_name="X", specialty="Emergency")


def _dka(severity=3):
    return {
        "name": "Diabetic Ketoacidosis", "aliases": ["dka"], "specialties": ["Emergency"],
        "severity": severity, "vitals": {}, "lab_deviations": {}, "differentials": ["Sepsis"],
        "appropriate_tests": ["BMP", "ABG"], "first_line_drugs": ["Insulin (regular)", "Normal Saline"],
        "contraindicated_drugs": [], "guideline": {"url": "https://example.org/dka"},
        "safety_gates": [
            {"id": "g0", "type": "test_before_drug", "test": "BMP", "drug": "Insulin (regular)",
             "rationale": "Check potassium before insulin.", "penalty": 15},
            {"id": "g1", "type": "test_before_test", "first": "Beta-hCG", "then": "CT Abdomen/Pelvis",
             "rationale": "Confirm pregnancy before imaging.", "penalty": 10},
        ],
    }


def _keys(st):
    return [e.action for e in st.score_log]


def _delta(st, action):
    return sum(e.delta for e in st.score_log if e.action == action)


# --- severity multiplier -------------------------------------------------- #
def test_severity_multiplier():
    assert safety.severity_multiplier({"severity": 1}) == 1.0
    assert safety.severity_multiplier({"severity": 2}) == 1.5
    assert safety.severity_multiplier({"severity": 3}) == 2.0
    assert safety.severity_multiplier({}) == 1.0          # unspecified → unchanged


def test_severity_scales_diagnosis_penalty_and_reward():
    dis = _dka(severity=3)
    st = _state(); st.ordered_tests.append("BMP")         # avoid premature-closure flag
    critic.submit_diagnosis(st, dis, "wrongdx")
    assert _delta(st, "incorrect_diagnosis") == -30       # -15 × 2.0

    st2 = _state(); st2.ordered_tests.append("BMP")
    critic.submit_diagnosis(st2, dis, "diabetic ketoacidosis")
    assert _delta(st2, "correct_diagnosis") == 60         # +30 × 2.0
    assert st2.status == "complete"


def test_no_severity_is_unscaled():
    dis = {"name": "Contact Dermatitis", "aliases": ["dermatitis"], "differentials": ["x"],
           "guideline": {}, "appropriate_tests": []}      # no severity
    st = _state()
    critic.submit_diagnosis(st, dis, "wrongdx")
    assert _delta(st, "incorrect_diagnosis") == -15


# --- executable gates ----------------------------------------------------- #
def test_test_before_drug_gate_violation_scaled():
    dis = _dka(severity=3)
    st = _state(); st.allergies = []
    msgs = clinical.prescribe_drug(st, dis, "Insulin (regular)")   # no BMP first
    assert "safety_gate_violation" in _keys(st)
    assert _delta(st, "safety_gate_violation") == -30              # 15 × 2.0
    assert any(m.role == "safety" for m in msgs)


def test_no_violation_when_prerequisite_ordered_first():
    dis = _dka()
    st = _state(); st.allergies = []
    st.ordered_tests.append("BMP")                                 # potassium checked first
    clinical.prescribe_drug(st, dis, "Insulin (regular)")
    assert "safety_gate_violation" not in _keys(st)


def test_test_before_test_gate_violation():
    dis = _dka()
    st = _state()
    clinical.order_test(st, dis, "CT Abdomen/Pelvis", RNG)         # imaging before Beta-hCG
    assert "safety_gate_violation" in _keys(st)


def test_gate_fires_once():
    dis = _dka()
    st = _state(); st.allergies = []
    clinical.prescribe_drug(st, dis, "Insulin (regular)")          # g0 fires
    again = safety.check_gates(st, dis, "prescribe_drug", "Insulin (regular)")
    assert again == []                                             # already penalised
    assert _keys(st).count("safety_gate_violation") == 1


def test_gate_violation_in_debrief_safety_flags():
    dis = _dka()
    st = _state(); st.allergies = []; st.status = "complete"
    clinical.prescribe_drug(st, dis, "Insulin (regular)")          # violation
    d = debrief.build_debrief(st, dis)
    assert any("Safety sequence" in s for s in d.safety_flags)

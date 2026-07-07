"""Integration tests: whole-case simulations through the engine (PLAN §8).

Assert that a guideline-concordant work-up scores well and that unsafe / low-value
actions are penalised, matching the expected clinical outcome.
"""
import pytest

from src import engine
from src.state import GameState
from src.store import SessionStore

PNEUMONIA = "C0032285"
DKA = "C0011880"
APPENDICITIS = "C0003615"


def _new(disease_id):
    """Fresh case with allergies cleared for deterministic drug scoring."""
    st = engine.create_case(["Internal Medicine", "Emergency", "Surgery"],
                            student_id="stu", disease_id=disease_id)
    st.allergies = []
    return st


def _actions(state, actions):
    for a in actions:
        engine.perform_action(state, a)


def _action_keys(state):
    return [e.action for e in state.score_log]


def test_points_equal_sum_of_log():
    """Invariant: the running total always equals the sum of all deltas."""
    st = _new(PNEUMONIA)
    _actions(st, [
        {"type": "order_test", "payload": "CBC"},
        {"type": "prescribe_drug", "payload": "Amoxicillin"},
        {"type": "order_test", "payload": "CT Abdomen/Pelvis"},
    ])
    assert st.points == sum(e.delta for e in st.score_log)


def test_good_pneumonia_workup_scores_and_completes():
    st = _new(PNEUMONIA)
    _actions(st, [
        {"type": "ask_history", "payload": "When did the cough start?"},
        {"type": "perform_exam", "payload": "Auscultate lungs"},
        {"type": "order_test", "payload": "Chest X-ray"},   # appropriate
        {"type": "order_test", "payload": "CBC"},           # appropriate
        {"type": "order_test", "payload": "CT Abdomen/Pelvis"},  # unnecessary
        {"type": "prescribe_drug", "payload": "Amoxicillin"},    # first-line
        {"type": "submit_diagnosis", "payload": "pneumonia"},    # correct
    ])
    keys = _action_keys(st)
    assert st.status == "complete"
    assert keys.count("order_appropriate_test") == 2
    assert "order_unnecessary_test" in keys
    assert "prescribe_guideline_first_line" in keys
    assert "correct_diagnosis" in keys
    assert st.points > 0


def test_unnecessary_test_is_penalised():
    st = _new(PNEUMONIA)
    engine.perform_action(st, {"type": "order_test", "payload": "CT Abdomen/Pelvis"})
    ev = st.score_log[-1]
    assert ev.action == "order_unnecessary_test"
    assert ev.delta < 0


def test_contraindicated_drug_penalised():
    st = _new(DKA)
    engine.perform_action(st, {"type": "prescribe_drug", "payload": "Metformin"})
    actions = _action_keys(st)
    assert "prescribe_contraindicated_drug" in actions
    assert st.points < 0


def test_first_line_dka_fluids_rewarded():
    st = _new(DKA)
    engine.perform_action(st, {"type": "prescribe_drug", "payload": "Normal Saline"})
    assert "prescribe_guideline_first_line" in _action_keys(st)
    assert st.points > 0


def test_indicated_surgery_rewarded():
    st = _new(APPENDICITIS)
    engine.perform_action(st, {"type": "request_surgery", "payload": "Appendectomy"})
    assert "emergency_surgery_indicated" in _action_keys(st)
    assert st.points > 0


def test_unwarranted_surgery_penalised():
    st = _new(PNEUMONIA)
    engine.perform_action(st, {"type": "request_surgery", "payload": "Appendectomy"})
    assert "unwarranted_surgery" in _action_keys(st)
    assert st.points < 0


def test_retries_exhaust_and_fail_case():
    st = _new(PNEUMONIA)
    for _ in range(st.max_retries):
        engine.perform_action(st, {"type": "submit_diagnosis", "payload": "lupus"})
    assert st.status == "failed"
    assert _action_keys(st).count("incorrect_diagnosis") == st.max_retries
    # further actions are blocked once the case is closed
    before = st.turn
    engine.perform_action(st, {"type": "order_test", "payload": "CBC"})
    assert st.turn == before


def test_actions_blocked_after_completion():
    st = _new(PNEUMONIA)
    engine.perform_action(st, {"type": "submit_diagnosis", "payload": "pneumonia"})
    assert st.status == "complete"
    n = len(st.score_log)
    engine.perform_action(st, {"type": "order_test", "payload": "CBC"})
    assert len(st.score_log) == n  # nothing scored


def test_store_persistence_across_actions():
    store = SessionStore(redis_url="")
    st = engine.create_case(["Internal Medicine"], student_id="stu",
                            disease_id=PNEUMONIA, store=store)
    engine.perform_action(st, {"type": "order_test", "payload": "CBC"}, store=store)
    reloaded = store.load(st.redis_key())
    assert reloaded is not None
    assert reloaded.ordered_tests == ["CBC"]
    assert reloaded.points == st.points

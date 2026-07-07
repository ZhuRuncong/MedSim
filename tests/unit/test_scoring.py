"""Unit tests for the rubric / scoring engine (PLAN §8)."""
from src import config
from src.scoring import apply_points
from src.state import GameState


def _state():
    return GameState(case_id="t", disease_id="x", disease_name="X", specialty="Internal Medicine")


def test_apply_points_uses_rubric():
    st = _state()
    delta = apply_points(st, "order_appropriate_test", "ClinicalAgent")
    assert delta == config.POINTS["order_appropriate_test"] == 10
    assert st.points == 10
    assert len(st.score_log) == 1
    ev = st.score_log[0]
    assert ev.action == "order_appropriate_test" and ev.delta == 10
    assert ev.agent == "ClinicalAgent"


def test_apply_points_explicit_delta_overrides():
    st = _state()
    delta = apply_points(st, "major_drug_interaction", "ClinicalAgent", delta=-5)
    assert delta == -5 and st.points == -5


def test_penalties_accumulate():
    st = _state()
    apply_points(st, "correct_diagnosis", "CriticAgent")     # +30
    apply_points(st, "unwarranted_surgery", "SurgeryAgent")  # -25
    assert st.points == 5
    assert [e.action for e in st.score_log] == ["correct_diagnosis", "unwarranted_surgery"]


def test_level_tracks_points():
    st = _state()
    assert st.level == 1
    apply_points(st, "correct_diagnosis", "CriticAgent", delta=250)
    assert st.level == config.level_for_points(250) == 3


def test_all_agent_rubric_keys_exist():
    # Guard against typos: keys used by agents must be defined in POINTS
    # (or passed with an explicit delta). These are the rubric-driven ones.
    required = {
        "order_appropriate_test", "order_unnecessary_test", "duplicate_action",
        "prescribe_guideline_first_line", "prescribe_contraindicated_drug",
        "prescribe_allergen", "prescribe_reasonable_drug", "avoid_allergy",
        "correct_diagnosis", "partial_diagnosis", "incorrect_diagnosis",
        "emergency_surgery_indicated", "unwarranted_surgery",
        "use_knowledge_query", "informative_history", "informative_exam",
    }
    assert required <= set(config.POINTS)

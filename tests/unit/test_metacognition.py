"""Metacognition scoring: differential ranking, confidence calibration, premature closure."""
from src import data_loader
from src.agents import critic
from src.state import GameState

PNEUMONIA = "C0032285"


def _state():
    dis = data_loader.get_disease(PNEUMONIA)
    st = GameState(case_id="t", disease_id=PNEUMONIA, disease_name=dis["name"],
                   specialty="Internal Medicine")
    return st, dis


def _keys(st):
    return [e.action for e in st.score_log]


def _delta(st, action):
    return sum(e.delta for e in st.score_log if e.action == action)


def _worked_up(st):
    st.ordered_tests.append("CBC")  # avoid the premature-closure flag


# --- differential ranking ------------------------------------------------- #
def test_true_dx_in_differentials_earns_rank_credit():
    st, dis = _state()
    _worked_up(st)
    critic.submit_diagnosis(st, dis, "acute bronchitis", differentials=["influenza", "pneumonia"])
    assert "differential_recognition" in _keys(st)
    assert _delta(st, "differential_recognition") == 8   # rank #2
    assert "incorrect_diagnosis" not in _keys(st)
    assert st.status == "active"  # wrong primary still consumes a retry
    assert st.retries == 1


def test_rank1_differential_credit():
    st, dis = _state()
    _worked_up(st)
    critic.submit_diagnosis(st, dis, "bronchitis", differentials=["pneumonia", "asthma"])
    assert _delta(st, "differential_recognition") == 12  # rank #1


def test_correct_primary_ignores_differentials_and_wins():
    st, dis = _state()
    _worked_up(st)
    critic.submit_diagnosis(st, dis, "pneumonia", differentials=["bronchitis"])
    assert st.status == "complete"
    assert "correct_diagnosis" in _keys(st)


# --- confidence calibration ---------------------------------------------- #
def test_calibrated_confidence_bonus_when_correct_and_sure():
    st, dis = _state()
    _worked_up(st)
    critic.submit_diagnosis(st, dis, "pneumonia", confidence=90)
    assert "calibrated_confidence" in _keys(st) and _delta(st, "calibrated_confidence") == 5


def test_overconfident_error_penalised():
    st, dis = _state()
    _worked_up(st)
    critic.submit_diagnosis(st, dis, "lupus", confidence=95)
    assert "overconfident_error" in _keys(st) and _delta(st, "overconfident_error") == -8


def test_prudent_uncertainty_rewarded():
    st, dis = _state()
    _worked_up(st)
    critic.submit_diagnosis(st, dis, "lupus", confidence=15)
    assert "prudent_uncertainty" in _keys(st) and _delta(st, "prudent_uncertainty") == 2


def test_neutral_confidence_no_calibration_event():
    st, dis = _state()
    _worked_up(st)
    critic.submit_diagnosis(st, dis, "pneumonia", confidence=50)
    assert "calibrated_confidence" not in _keys(st)
    assert "overconfident_error" not in _keys(st)


# --- premature closure ---------------------------------------------------- #
def test_premature_closure_when_no_workup():
    st, dis = _state()  # no tests ordered, no exams performed
    critic.submit_diagnosis(st, dis, "lupus")
    assert "premature_closure" in _keys(st)


def test_no_premature_closure_after_workup():
    st, dis = _state()
    st.ordered_tests.append("CBC")
    critic.submit_diagnosis(st, dis, "lupus")
    assert "premature_closure" not in _keys(st)


def test_no_premature_closure_on_partial():
    st, dis = _state()  # 'tuberculosis' is a listed differential of pneumonia
    critic.submit_diagnosis(st, dis, "tuberculosis")
    assert "partial_diagnosis" in _keys(st)
    assert "premature_closure" not in _keys(st)  # a plausible differential isn't premature


# --- invariants & backward-compat ---------------------------------------- #
def test_points_invariant_with_metacognition():
    st, dis = _state()
    critic.submit_diagnosis(st, dis, "lupus", differentials=["sarcoidosis"], confidence=88)
    assert st.points == sum(e.delta for e in st.score_log)


def test_backward_compatible_plain_diagnosis():
    st, dis = _state()
    st.ordered_tests.append("CBC")
    msgs = critic.submit_diagnosis(st, dis, "pneumonia")  # no differentials/confidence
    assert st.status == "complete"
    assert "calibrated_confidence" not in _keys(st)  # no confidence → no calibration
    assert msgs and msgs[0].kind == "success"

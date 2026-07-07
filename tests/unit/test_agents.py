"""Agent-level behaviour: allergy-aware prescribing, duplicate guards, LLM patient."""
from src import data_loader, engine
from src.agents import clinical, knowledge, patient, supervisor
from src.llm import MockClient
from src.state import GameState

PNEUMONIA = "C0032285"


def _state(disease_id=PNEUMONIA, allergies=None):
    dis = data_loader.get_disease(disease_id)
    st = GameState(case_id="t", disease_id=disease_id, disease_name=dis["name"],
                   specialty="Internal Medicine")
    st.allergies = allergies or []
    return st, dis


def _keys(st):
    return [e.action for e in st.score_log]


def test_avoid_allergy_awarded_only_when_navigated():
    # Penicillin-allergic pneumonia: choosing a safe macrolide navigates the
    # allergy (Amoxicillin, a first-line option, would have conflicted).
    st, dis = _state(allergies=["penicillin"])
    clinical.prescribe_drug(st, dis, "Azithromycin")
    assert "prescribe_guideline_first_line" in _keys(st)
    assert "avoid_allergy" in _keys(st)


def test_avoid_allergy_not_awarded_for_unrelated_allergy():
    # An opioid allergy is irrelevant to amoxicillin — no bonus.
    st, dis = _state(allergies=["opioid"])
    clinical.prescribe_drug(st, dis, "Amoxicillin")
    assert "prescribe_guideline_first_line" in _keys(st)
    assert "avoid_allergy" not in _keys(st)


def test_cross_reactivity_no_bonus_and_surfaces_caution():
    # Penicillin-allergic patient given a cephalosporin (Ceftriaxone): first-line,
    # but NO avoid_allergy bonus, and the cross-reactivity caution is shown.
    st, dis = _state(allergies=["penicillin"])
    msgs = clinical.prescribe_drug(st, dis, "Ceftriaxone")
    assert "avoid_allergy" not in _keys(st)
    assert any("caution" in m.text.lower() for m in msgs)


def test_ask_history_duplicate_not_re_rewarded():
    st, dis = _state()
    supervisor.handle_history(st, dis, "Do you have any allergies?")
    p1 = st.points
    supervisor.handle_history(st, dis, "Do you have any allergies?")  # repeat
    assert "duplicate_action" in _keys(st)
    assert st.points < p1  # penalised, not rewarded again


def test_perform_exam_duplicate_penalised():
    st, dis = _state()
    supervisor.handle_exam(st, dis, "Auscultate lungs")
    p1 = st.points
    supervisor.handle_exam(st, dis, "Auscultate lungs")
    assert "duplicate_action" in _keys(st)
    assert st.points < p1


# --- grounded LLM patient dialogue ---------------------------------------- #
def _populated_state():
    st = engine.create_case(["Internal Medicine"], disease_id=PNEUMONIA)
    st.allergies = []
    return st, data_loader.get_disease(PNEUMONIA)


def test_patient_llm_rephrases_but_keeps_deterministic_scoring():
    st, dis = _populated_state()
    mock = MockClient([{"reply": "It's been a rough three days with this deep, nasty cough."}])
    res = patient.answer_history(st, dis, "When did the cough start?", client=mock)
    assert res["answer"] == "It's been a rough three days with this deep, nasty cough."
    assert res.get("llm") is True
    assert res["informative"] is True  # scoring flag still decided deterministically


def test_patient_llm_leak_guard_catches_alias():
    st, dis = _populated_state()
    # Alias-only leak ("pneumonia", not the full name) must still be rejected.
    mock = MockClient([{"reply": "I think I have pneumonia, doctor."}])
    res = patient.answer_history(st, dis, "When did the cough start?", client=mock)
    assert "pneumonia" not in res["answer"].lower()
    assert not res.get("llm")


def test_patient_llm_leak_guard_allows_ordinary_words():
    st, dis = _populated_state()
    # 'chest' appears in an alias ('chest infection') but must NOT be blocked.
    mock = MockClient([{"reply": "My chest hurts when I take a deep breath."}])
    res = patient.answer_history(st, dis, "Any chest pain?", client=mock)
    assert res.get("llm") is True
    assert res["answer"] == "My chest hurts when I take a deep breath."


def test_patient_llm_error_falls_back_to_deterministic():
    st, dis = _populated_state()
    broken = MockClient([])  # complete_json raises -> caught
    res = patient.answer_history(st, dis, "Do you have any allergies?", client=broken)
    assert res["answer"] == "No known drug allergies."  # deterministic
    assert not res.get("llm")


def test_patient_deterministic_when_no_client():
    st, dis = _populated_state()
    res = patient.answer_history(st, dis, "When did the cough start?")  # no key configured
    assert not res.get("llm")
    assert res["informative"] is True


# --- grounded (RAG) KnowledgeAgent ---------------------------------------- #
GROUNDED_Q = "first-line treatment for community-acquired pneumonia?"


def test_knowledge_llm_synthesizes_grounded_and_scores():
    st, _ = _state()
    synth = "For community-acquired pneumonia, amoxicillin or a macrolide is first-line."
    mock = MockClient([{"answer": synth}])
    msgs = knowledge.ask(st, GROUNDED_Q, client=mock)
    assert msgs[0].text == synth
    assert msgs[0].citations                     # citations preserved from retrieval
    assert "use_knowledge_query" in _keys(st)    # scoring unchanged (+2)


def test_knowledge_llm_skipped_when_no_source():
    st, _ = _state()
    mock = MockClient([{"answer": "should not be used"}])
    msgs = knowledge.ask(st, "what is the capital of France", client=mock)
    assert mock.calls == []                       # LLM not invoked without a grounded source
    assert "authoritative" in msgs[0].text.lower()
    assert "use_knowledge_query" not in _keys(st)  # no reward


def test_knowledge_llm_error_falls_back_to_retrieved_text():
    st, _ = _state()
    broken = MockClient([])                        # complete_json raises -> caught
    msgs = knowledge.ask(st, GROUNDED_Q, client=broken)
    assert "amoxicillin" in msgs[0].text.lower()   # deterministic retrieved answer
    assert "use_knowledge_query" in _keys(st)


def test_knowledge_deterministic_when_no_client():
    st, _ = _state()
    msgs = knowledge.ask(st, GROUNDED_Q)           # no key configured
    assert msgs[0].citations
    assert "use_knowledge_query" in _keys(st)

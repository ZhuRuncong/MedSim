"""End-of-case DebriefAgent: deterministic diff + optional grounded LLM voice."""
from src import data_loader, engine
from src.agents import debrief
from src.llm import MockClient

PNEUMONIA = "C0032285"
DKA = "C0011880"


def _play(disease_id, actions, allergies=None):
    st = engine.create_case(["Internal Medicine", "Emergency"], disease_id=disease_id)
    st.allergies = allergies or []
    for a in actions:
        engine.perform_action(st, a)
    return st


def test_debrief_built_on_completion():
    st = _play(PNEUMONIA, [
        {"type": "order_test", "payload": "Chest X-ray"},
        {"type": "order_test", "payload": "CBC"},
        {"type": "prescribe_drug", "payload": "Amoxicillin"},
        {"type": "submit_diagnosis", "payload": "pneumonia"},
    ])
    d = st.debrief
    assert d is not None
    assert d.outcome == "correct"
    assert d.true_diagnosis == "Community-Acquired Pneumonia"
    assert "Chest X-ray" in d.tests_hit and "CBC" in d.tests_hit
    assert "Amoxicillin" in d.drugs_first_line
    assert d.efficiency > 0 and d.max_points > 0
    assert d.citation and d.attending_note
    assert d.teaching_points


def test_debrief_flags_missed_and_low_value():
    st = _play(PNEUMONIA, [
        {"type": "order_test", "payload": "CT Abdomen/Pelvis"},  # low-value for pneumonia
        {"type": "submit_diagnosis", "payload": "pneumonia"},
    ])
    d = st.debrief
    assert "CT Abdomen/Pelvis" in d.tests_low_value
    assert d.tests_missed  # never ordered the appropriate ones
    assert d.drugs_missed_first_line  # never treated


def test_debrief_flags_harmful_drug():
    # DKA: Metformin is contraindicated
    st = _play(DKA, [
        {"type": "prescribe_drug", "payload": "Metformin"},
        {"type": "submit_diagnosis", "payload": "wrongdx"},
        {"type": "submit_diagnosis", "payload": "wrongdx2"},
        {"type": "submit_diagnosis", "payload": "wrongdx3"},  # exhaust retries -> failed
    ])
    d = st.debrief
    assert d.outcome == "missed"
    assert "Metformin" in d.drugs_harmful
    assert any("Metformin" in s for s in d.safety_flags)


def test_debrief_built_only_once_and_persists():
    st = _play(PNEUMONIA, [{"type": "submit_diagnosis", "payload": "pneumonia"}])
    first = st.debrief
    # further (blocked) actions must not rebuild it
    engine.perform_action(st, {"type": "order_test", "payload": "CBC"})
    assert st.debrief is first


def test_debrief_llm_note_is_grounded_and_optional():
    dis = data_loader.get_disease(PNEUMONIA)
    st = _play(PNEUMONIA, [{"type": "submit_diagnosis", "payload": "pneumonia"}])
    # deterministic note by default
    assert not st.debrief.llm and st.debrief.attending_note

    # rebuilding with an injected client yields the LLM's (grounded) text
    mock = MockClient([{"note": "Solid work — you nailed the pneumonia and treated it correctly."}])
    d = debrief.build_debrief(st, dis, client=mock)
    assert d.llm and d.attending_note.startswith("Solid work")


def test_debrief_survives_store_round_trip():
    from src.store import SessionStore
    store = SessionStore(redis_url="")
    st = _play(PNEUMONIA, [{"type": "submit_diagnosis", "payload": "pneumonia"}])
    store.save(st)
    loaded = store.load(st.redis_key())
    assert loaded.debrief is not None
    assert loaded.debrief.true_diagnosis == "Community-Acquired Pneumonia"

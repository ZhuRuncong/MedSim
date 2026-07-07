"""Agent-trajectory tracing + LangGraph/router parity tests."""
import pytest

from src import data_loader, engine, trace
from src.llm import MockClient
from src.store import SessionStore

PNEUMONIA = "C0032285"

SCRIPT = [
    {"type": "ask_history", "payload": "When did the cough start?"},
    {"type": "order_test", "payload": "Chest X-ray"},
    {"type": "prescribe_drug", "payload": "Amoxicillin"},
    {"type": "knowledge_query", "payload": "first-line treatment for community-acquired pneumonia?"},
    {"type": "submit_diagnosis", "payload": "pneumonia"},
]


def _run_script(monkeypatch=None, use_graph="1"):
    if monkeypatch is not None:
        monkeypatch.setenv("MEDSIM_USE_LANGGRAPH", use_graph)
        engine._compiled_graph = None  # force re-resolution of the path
    st = engine.create_case(["Internal Medicine"], disease_id=PNEUMONIA)
    st.allergies = []
    for a in SCRIPT:
        engine.perform_action(st, a)
    return st


def test_trace_records_every_action():
    st = _run_script()
    assert len(st.trace) == len(SCRIPT)
    by_action = {e.action: e for e in st.trace}
    assert by_action["ask_history"].agent == "PatientAgent"
    assert by_action["order_test"].tools == ["lab_simulator"]
    assert by_action["prescribe_drug"].tools == ["drug_effect_engine"]
    assert by_action["knowledge_query"].agent == "KnowledgeAgent"
    assert by_action["submit_diagnosis"].agent == "CriticAgent"
    assert all(e.router == "SupervisorAgent" for e in st.trace)
    assert all(e.duration_ms >= 0 for e in st.trace)


def test_trace_points_deltas_match_scoring():
    st = _run_script()
    assert sum(e.points_delta for e in st.trace) == st.points


def test_trace_llm_flag_set_by_llm_voiced_reply():
    st = engine.create_case(["Internal Medicine"], disease_id=PNEUMONIA)
    st.allergies = []
    from src.agents import patient
    mock = MockClient([{"reply": "About three days now, and it keeps getting worse."}])
    res = patient.answer_history(st, data_loader.get_disease(PNEUMONIA),
                                 "When did it start?", client=mock)
    assert res.get("llm") is True


def test_trace_survives_store_round_trip():
    store = SessionStore(redis_url="")
    st = _run_script()
    store.save(st)
    loaded = store.load(st.redis_key())
    assert len(loaded.trace) == len(SCRIPT)
    assert loaded.trace[0].agent == "PatientAgent"


def test_trace_summary_and_otel_export():
    st = _run_script()
    s = trace.summary(st.trace)
    assert s["actions"] == len(SCRIPT)
    assert "ClinicalAgent" in s["agents"] and "CriticAgent" in s["agents"]
    assert "lab_simulator" in s["tools"]
    span = st.trace[1].to_otel_dict()
    assert span["name"] == "medsim.action.order_test"
    assert span["attributes"]["medsim.agent"] == "ClinicalAgent"


# --- LangGraph vs deterministic-router parity ------------------------------ #
langgraph = pytest.importorskip("langgraph")


def test_graph_and_router_paths_are_equivalent(monkeypatch):
    st_router = _run_script(monkeypatch, use_graph="0")
    st_graph = _run_script(monkeypatch, use_graph="1")
    engine._compiled_graph = None

    assert st_router.points == st_graph.points
    assert st_router.status == st_graph.status == "complete"
    assert [e.action for e in st_router.score_log] == [e.action for e in st_graph.score_log]
    assert [m.kind for m in st_router.feed] == [m.kind for m in st_graph.feed]

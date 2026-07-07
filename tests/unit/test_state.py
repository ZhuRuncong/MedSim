"""GameState model + session store round-trip."""
from src.state import GameState, Message, ScoreEvent
from src.store import SessionStore


def test_gamestate_defaults():
    st = GameState(case_id="c1", disease_id="d1", disease_name="D", specialty="Emergency")
    assert st.points == 0 and st.turn == 0 and st.status == "active"
    assert st.max_retries == 3
    assert st.redis_key() == "session:guest:c1"


def test_gamestate_json_round_trip():
    st = GameState(case_id="c1", disease_id="d1", disease_name="D", specialty="Emergency")
    st.feed.append(Message(role="patient", text="hi", points_delta=1))
    st.score_log.append(ScoreEvent(action="x", delta=5, agent="A"))
    payload = st.model_dump_json()
    st2 = GameState.model_validate_json(payload)
    assert st2.feed[0].text == "hi"
    assert st2.score_log[0].delta == 5


def test_store_memory_backend_round_trip():
    store = SessionStore(redis_url="")  # force memory backend
    assert store.backend == "memory"
    st = GameState(case_id="abc", disease_id="d", disease_name="D",
                   specialty="Surgery", student_id="stu")
    st.points = 42
    store.save(st)
    loaded = store.load(st.redis_key())
    assert loaded is not None and loaded.points == 42
    store.delete(st.redis_key())
    assert store.load(st.redis_key()) is None

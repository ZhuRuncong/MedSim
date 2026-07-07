"""A persisted generated case must survive an in-memory registry wipe.

Repro of the reported bug: Streamlit hot-reloads on a source edit, which clears
``data_loader._REGISTRY`` (module global) but keeps ``st.session_state`` (the
active case). Acting on that case then failed with "unknown disease 'GEN-...'".
The engine now self-heals by reloading the on-disk generated cache on a miss.
"""
import copy

from src import config, data_loader, engine


def test_perform_action_reloads_generated_case_after_registry_wipe(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GENERATED_DIR", tmp_path)

    # Mint a "generated" case from a fixture's shape, register it, start a case,
    # and persist it to the (temp) generated cache.
    gen = copy.deepcopy(data_loader.get_disease("C0155626"))
    gen["id"], gen["generated"] = "GEN-selfheal", True
    data_loader.register_disease(gen)
    try:
        state = engine.create_case(["Emergency"], disease_id="GEN-selfheal")
        data_loader.persist_generated(gen)

        # Simulate a hot-reload: registry wiped, but the load-guard flag is stale.
        data_loader._REGISTRY.pop("GEN-selfheal", None)
        if "GEN-selfheal" in data_loader._ORDER:
            data_loader._ORDER.remove("GEN-selfheal")
        monkeypatch.setattr(data_loader, "_GENERATED_LOADED", True)
        assert data_loader.get_disease("GEN-selfheal") is None

        # The action must NOT raise — it reloads the case from disk and proceeds.
        msgs = engine.perform_action(state, {"type": "ask_history", "payload": "onset"})
        assert msgs
        assert data_loader.get_disease("GEN-selfheal") is not None
    finally:
        data_loader._REGISTRY.pop("GEN-selfheal", None)
        if "GEN-selfheal" in data_loader._ORDER:
            data_loader._ORDER.remove("GEN-selfheal")

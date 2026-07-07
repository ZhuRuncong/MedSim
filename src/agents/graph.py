"""LangGraph workflow (PLAN §5) — optional.

The running app uses the deterministic router in ``engine.perform_action`` (which
is exactly equivalent and needs no extra dependency). This module wires the same
agent functions into a real ``langgraph`` StateGraph for those who install it,
enforcing the turn-based topology from PLAN's Mermaid diagram.

Import is always safe: if ``langgraph`` is missing, ``LANGGRAPH_AVAILABLE`` is
False and ``build_graph`` raises a clear error instead of crashing at import.
"""
from __future__ import annotations


from typing import Any, Dict, List, TypedDict

from ..state import GameState, Message
from . import clinical, critic, knowledge, supervisor, surgery

try:  # pragma: no cover - exercised only when langgraph is installed
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False
    END = "__end__"


class GraphState(TypedDict, total=False):
    """LangGraph channel schema (last-value-wins on every channel)."""

    game_state: GameState
    disease: dict
    action: dict
    rng: Any
    route: str
    messages: List[Message]

# Re-export the canonical diagram.
from ..engine import MERMAID  # noqa: E402

_ROUTE = {
    "ask_history": "patient",
    "perform_exam": "patient",
    "order_test": "clinical",
    "prescribe_drug": "clinical",
    "request_surgery": "surgery",
    "knowledge_query": "knowledge",
    "submit_diagnosis": "critic",
}


# --------------------------------------------------------------------------- #
# Nodes (operate on a plain dict channel)
# --------------------------------------------------------------------------- #
def _supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gs = state["game_state"]
    action = state["action"]
    # Mirror supervisor.route exactly: don't advance the turn for a closed case
    # or an unknown action type.
    if gs.status != "active":
        return {"route": "END",
                "messages": [Message(role="supervisor",
                                     text="This case is closed.", kind="info")]}
    if action.get("type") not in _ROUTE:
        return {"route": "END",
                "messages": [Message(role="supervisor",
                                     text=f"Unknown action '{action.get('type')}'.",
                                     kind="info")]}
    gs.turn += 1
    return {"route": _ROUTE[action["type"]]}


def _patient_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gs, disease, action = state["game_state"], state["disease"], state["action"]
    payload = str(action.get("payload") or "")
    if action["type"] == "ask_history":
        return {"messages": supervisor.handle_history(gs, disease, payload)}
    return {"messages": supervisor.handle_exam(gs, disease, payload)}


def _clinical_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gs, disease, action = state["game_state"], state["disease"], state["action"]
    if action["type"] == "order_test":
        msgs = clinical.order_test(gs, disease, str(action["payload"]), state["rng"])
    else:
        msgs = clinical.prescribe_drug(gs, disease, str(action["payload"]))
    return {"messages": msgs}


def _surgery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gs, disease, action = state["game_state"], state["disease"], state["action"]
    return {"messages": surgery.request_surgery(gs, disease, str(action["payload"]), state["rng"])}


def _knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gs, action = state["game_state"], state["action"]
    return {"messages": knowledge.ask(gs, str(action["payload"]))}


def _critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gs, disease, action = state["game_state"], state["disease"], state["action"]
    return {"messages": critic.submit_diagnosis(
        gs, disease, str(action["payload"]),
        differentials=action.get("differentials"), confidence=action.get("confidence"))}


def build_graph():
    """Compile and return the LangGraph workflow (requires ``langgraph``)."""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "langgraph is not installed. `pip install langgraph` to use the graph, "
            "or use engine.perform_action() which is equivalent."
        )
    sg = StateGraph(GraphState)
    sg.add_node("supervisor", _supervisor_node)
    sg.add_node("patient", _patient_node)
    sg.add_node("clinical", _clinical_node)
    sg.add_node("surgery", _surgery_node)
    sg.add_node("knowledge", _knowledge_node)
    sg.add_node("critic", _critic_node)

    sg.set_entry_point("supervisor")
    sg.add_conditional_edges(
        "supervisor",
        lambda s: s.get("route", "END"),
        {"patient": "patient", "clinical": "clinical", "surgery": "surgery",
         "knowledge": "knowledge", "critic": "critic", "END": END},
    )
    for node in ("patient", "clinical", "surgery", "knowledge", "critic"):
        sg.add_edge(node, END)
    return sg.compile()

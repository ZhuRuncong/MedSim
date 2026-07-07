"""Agent trajectory observability.

Every student action produces a :class:`TraceEvent` capturing which agents ran,
which tools they called, what changed (points, state) and how long it took —
the "agent trajectory" a grader or evaluation harness can replay and audit.

This is deliberately OpenTelemetry-*shaped* (span-like events with name,
attributes, duration) while staying dependency-free; `to_otel_dict()` emits a
structure that maps 1:1 onto OTel spans if an exporter is wired up later.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Which specialised agent (and tools) the supervisor routes each action to.
ROUTING: Dict[str, dict] = {
    "ask_history": {"agent": "PatientAgent", "tools": []},
    "perform_exam": {"agent": "PatientAgent", "tools": []},
    "order_test": {"agent": "ClinicalAgent", "tools": ["lab_simulator"]},
    "prescribe_drug": {"agent": "ClinicalAgent", "tools": ["drug_effect_engine"]},
    "request_surgery": {"agent": "SurgeryAgent", "tools": ["emergency_surgery"]},
    "knowledge_query": {"agent": "KnowledgeAgent", "tools": ["knowledge_lookup"]},
    "submit_diagnosis": {"agent": "CriticAgent", "tools": []},
}


class TraceEvent(BaseModel):
    """One span in the agent trajectory."""

    turn: int
    action: str                                  # student action type
    payload: str = ""                            # what was asked/ordered
    router: str = "SupervisorAgent"
    agent: str = ""                              # specialised agent that handled it
    tools: List[str] = Field(default_factory=list)
    llm_used: bool = False                       # did an LLM voice/synthesise output?
    points_before: int = 0
    points_after: int = 0
    messages: int = 0                            # feed messages produced
    duration_ms: float = 0.0
    status: str = "ok"                           # ok | blocked | error

    @property
    def points_delta(self) -> int:
        return self.points_after - self.points_before

    def to_otel_dict(self) -> dict:
        """Span-shaped export (name + attributes), OTel-compatible."""
        return {
            "name": f"medsim.action.{self.action}",
            "attributes": {
                "medsim.turn": self.turn,
                "medsim.router": self.router,
                "medsim.agent": self.agent,
                "medsim.tools": ",".join(self.tools),
                "medsim.payload": self.payload[:120],
                "medsim.llm_used": self.llm_used,
                "medsim.points_delta": self.points_delta,
                "medsim.status": self.status,
            },
            "duration_ms": self.duration_ms,
        }


@contextmanager
def record(state, action: dict):
    """Context manager used by the engine to trace one action end-to-end."""
    atype = str(action.get("type", "unknown"))
    route = ROUTING.get(atype, {"agent": "", "tools": []})
    ev = TraceEvent(
        turn=state.turn + 1,
        action=atype,
        payload=str(action.get("payload") or "")[:200],
        agent=route["agent"],
        tools=list(route["tools"]),
        points_before=state.points,
    )
    start = time.perf_counter()
    feed_before = len(state.feed)
    try:
        yield ev
    except Exception:
        ev.status = "error"
        raise
    finally:
        ev.duration_ms = round((time.perf_counter() - start) * 1000, 2)
        ev.points_after = state.points
        ev.turn = state.turn
        ev.messages = max(0, len(state.feed) - feed_before)
        if state.status != "active" and ev.action != "submit_diagnosis":
            # action arrived after the case closed and was refused by the router
            ev.status = "blocked" if ev.points_after == ev.points_before else ev.status
        state.trace.append(ev)


def summary(trace: List[TraceEvent]) -> dict:
    """Aggregate stats for a trajectory — used by the eval harness & notebook."""
    if not trace:
        return {"actions": 0, "agents": [], "tools": [], "llm_actions": 0,
                "points_delta": 0, "total_ms": 0.0}
    agents = sorted({e.agent for e in trace if e.agent})
    tools = sorted({t for e in trace for t in e.tools})
    return {
        "actions": len(trace),
        "agents": agents,
        "tools": tools,
        "llm_actions": sum(1 for e in trace if e.llm_used),
        "points_delta": trace[-1].points_after - trace[0].points_before,
        "total_ms": round(sum(e.duration_ms for e in trace), 2),
    }

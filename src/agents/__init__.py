"""LangGraph agents (PLAN §5).

Each agent is a set of stateless functions that receive the current GameState
(and the resolved disease dict) and return an updated state plus UI messages.
The SupervisorAgent routes between them; `graph.py` wires them into a true
LangGraph workflow when the library is installed.
"""
from . import clinical, critic, knowledge, patient, supervisor, surgery

__all__ = ["patient", "supervisor", "clinical", "surgery", "knowledge", "critic"]

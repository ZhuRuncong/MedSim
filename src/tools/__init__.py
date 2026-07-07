"""FastMCP custom tools (PLAN §3).

Each tool is a *pure function* implementing the clinical logic. `server.py`
wraps them as FastAPI endpoints, but agents and tests call the functions
directly so nothing requires a running server.
"""
from .symptom_generator import generate_symptoms
from .lab_simulator import simulate_labs
from .drug_effect_engine import evaluate_drug
from .knowledge_lookup import lookup
from .emergency_surgery import evaluate_surgery

__all__ = [
    "generate_symptoms",
    "simulate_labs",
    "evaluate_drug",
    "lookup",
    "evaluate_surgery",
]

"""MedSim — gamified medical-student hospital simulation.

Local-first: every module degrades gracefully when optional infrastructure
(Redis, Qdrant, an LLM API key, Tavily) is absent, so the full case loop runs
offline and deterministically.
"""

__version__ = "0.1.0"

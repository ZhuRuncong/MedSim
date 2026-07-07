"""KnowledgeAgent (PLAN §5): answers factual queries with citations.

Retrieves an authoritative source with the knowledge_lookup tool (local guideline
DB → Tavily fallback) and, when an LLM is configured, synthesizes a direct answer
*grounded strictly in that retrieved source* — retrieval-augmented generation, so
it can't fabricate facts or citations. Rewards a small number of points for
consulting an evidence source, per the rubric.
"""
from __future__ import annotations

from typing import List, Optional

from .. import config
from ..scoring import apply_points
from ..state import GameState, Message
from ..tools import lookup

AGENT = "KnowledgeAgent"

_KNOWLEDGE_SYSTEM = (
    "You are a clinical knowledge assistant for medical students. Answer the student's "
    "question using ONLY the provided source material — do not add drugs, doses, claims, "
    "or facts that the source does not support, and never invent citations. Be concise "
    "(1-4 sentences) and precise. If the source does not actually address the question, "
    "say what the source does cover and note the limitation. "
    'Respond as JSON: {"answer": "<your answer>"}.'
)


def _llm_synthesize(client, question: str, result: dict) -> str:
    cites = ", ".join(result.get("citations", [])) or "(none)"
    user = (f"SOURCE (retrieved from {result.get('source')}):\n{result.get('answer', '')}\n\n"
            f"CITATIONS: {cites}\n\nSTUDENT QUESTION: {question}\n\n"
            f"Answer the question using only the source above.")
    out = client.complete_json(_KNOWLEDGE_SYSTEM, user, temperature=0.3)
    return (out or {}).get("answer", "").strip()


def ask(state: GameState, question: str, client=None) -> List[Message]:
    result = lookup(question)
    grounded = result.get("source") not in (None, "none")
    answer = result["answer"]
    used_llm = False

    # Only reward — and only synthesize — when an authoritative source answered.
    delta = 0
    if grounded:
        delta = apply_points(state, "use_knowledge_query", AGENT,
                             reason=f"Consulted evidence source for: {question}")
        if client is None and config.knowledge_llm_enabled():
            from ..llm import get_client
            client = get_client()
        if client is not None:
            try:
                synth = _llm_synthesize(client, question, result)
                if synth:
                    answer = synth
                    used_llm = True
            except Exception:
                pass  # fall back to the retrieved source text

    return [Message(role="knowledge", text=answer, kind="info",
                    points_delta=delta, citations=result.get("citations", []),
                    turn=state.turn, llm=used_llm)]

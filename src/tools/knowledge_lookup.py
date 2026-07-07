"""knowledge_lookup — POST /api/v1/lookup.

Answer a factual query from the local guideline knowledge base (built from the
disease catalog + drug monographs). Falls back to the Tavily web API when it is
configured and the local DB has no confident match. Always returns citations.
PLAN §3.
"""
from __future__ import annotations

import re
from typing import List, Optional

from .. import config, data_loader

_TOKEN = re.compile(r"[a-z0-9\-]+")
_TREATMENT_HINTS = ("treat", "first-line", "first line", "manage", "therapy",
                    "antibiotic", "drug of choice", "medication")

# Common words carry no topical signal — exclude them from overlap scoring so a
# query like "what is the capital of France" doesn't match on "is/the/of".
_STOP = {"what", "is", "the", "of", "a", "an", "for", "to", "in", "on", "and",
         "or", "how", "do", "does", "with", "are", "be", "which", "that", "this",
         "i", "you", "it", "my", "at", "by", "as", "can", "should", "if", "when"}


def _tokens(text: str):
    return set(_TOKEN.findall(text.lower())) - _STOP


def _drug_answer(query: str) -> Optional[dict]:
    q = query.lower()
    for name, meta in data_loader.drug_db().get("drugs", {}).items():
        if meta.get("class") == "diagnostic":
            continue
        if name.lower() in q:
            answer = f"{name} — {meta.get('class')}. {meta.get('monograph', '')}"
            return {
                "query": query,
                "answer": answer,
                "source": "local:drug_monograph",
                "citations": [config.MEDSIM_API_URL + "/monograph/" + name.replace(" ", "%20")],
            }
    return None


def _name_hit(names, query_raw_tokens) -> bool:
    """True if a *specific* disease name/alias appears as whole words in the query.

    Whole-word (token-subset) matching avoids substring hits like 'cap' inside
    'capital'. We additionally require specificity — a multi-word name or a token
    of length >= 4 — so that short, ambiguous abbreviations ('mi', 'cap', 'acs',
    'dka', 'uti', 'rsv') don't confidently match casual queries like
    'mi casa is nice' or 'put a cap on spending'.
    """
    for n in names:
        ntok = set(_TOKEN.findall(n.lower()))
        if not ntok or not (ntok <= query_raw_tokens):
            continue
        if len(ntok) >= 2 or any(len(t) >= 4 for t in ntok):
            return True
    return False


def _disease_answer(query: str) -> Optional[dict]:
    q = query.lower()
    qtokens = _tokens(query)
    qtokens_raw = set(_TOKEN.findall(q))
    best = None
    best_score = 0.0
    for d in data_loader.diseases():
        names = [d["name"]] + d.get("aliases", [])
        name_hit = _name_hit(names, qtokens_raw)
        hay = " ".join(names + [d.get("teaching", "")] + d.get("differentials", []))
        overlap = len(qtokens & _tokens(hay))
        score = overlap + (5 if name_hit else 0)
        if score > best_score:
            best_score, best = score, d
    if best is None or best_score < 3:
        return None

    parts = [best.get("teaching", "")]
    if any(h in q for h in _TREATMENT_HINTS) and best.get("first_line_drugs"):
        parts.append("Guideline first-line options: "
                     + ", ".join(best["first_line_drugs"]) + ".")
    guideline = best.get("guideline", {})
    citations = [guideline.get("url")] if guideline.get("url") else []
    return {
        "query": query,
        "answer": " ".join(p for p in parts if p),
        "source": f"local:guideline:{guideline.get('name', 'clinical guideline')}",
        "citations": [c for c in citations if c],
    }


def _tavily_answer(query: str) -> Optional[dict]:
    if not config.USE_TAVILY:
        return None
    try:  # best-effort; never crash the case loop
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        res = client.search(query=query, max_results=3, include_answer=True)
        snippets = res.get("results", [])[:3]
        answer = res.get("answer") or " ".join(s.get("content", "")[:200] for s in snippets)
        return {
            "query": query,
            "answer": answer,
            "source": "tavily",
            "citations": [s.get("url") for s in snippets if s.get("url")],
        }
    except Exception:
        return None


def lookup(query: str, disease: Optional[dict] = None) -> dict:
    """Answer a factual query, preferring the local guideline DB."""
    for finder in (_disease_answer, _drug_answer):
        ans = finder(query)
        if ans:
            return ans
    ans = _tavily_answer(query)
    if ans:
        return ans
    return {
        "query": query,
        "answer": ("No authoritative local source matched that query. Try naming a "
                   "specific condition or drug (e.g. 'first-line treatment for "
                   "community-acquired pneumonia')."),
        "source": "none",
        "citations": [],
    }

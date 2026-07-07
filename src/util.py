"""Small shared helpers."""
from __future__ import annotations

import hashlib
import re
from typing import Iterable
from urllib.parse import unquote

import numpy as np


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from arbitrary parts (process-independent).

    Python's built-in ``hash`` is randomised per process; this is stable so a
    given case id always regenerates the same patient.
    """
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def case_rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def diagnosis_terms(disease: dict) -> set:
    """All strings that would give the diagnosis away: the name, every alias, and
    for any that carry a parenthetical (e.g. ``Acute MI (STEMI)``) both the
    paren-stripped form (``Acute MI``) and the parenthetical itself (``STEMI``).

    The expansion matters because the model may name a case one way and describe
    it another, and aliases aren't guaranteed to list every form.
    """
    raw = {disease.get("name", ""), *disease.get("aliases", [])}
    terms: set = set()
    for t in raw:
        t = (t or "").strip()
        if not t:
            continue
        terms.add(t)
        stripped = re.sub(r"\s*\([^)]*\)", "", t).strip()  # "Acute MI (STEMI)" -> "Acute MI"
        if stripped:
            terms.add(stripped)
        for inner in re.findall(r"\(([^)]*)\)", t):        # -> "STEMI"
            if inner.strip():
                terms.add(inner.strip())
    return terms


def redact_diagnosis(text: str, disease: dict, placeholder: str = "the underlying condition") -> str:
    """Scrub the hidden diagnosis (name + aliases + parenthetical forms) from
    student-facing text so feedback during an active case can't leak the answer.
    Whole-word, case-insensitive; longest terms first so multi-word names go
    before their abbreviations.
    """
    if not text:
        return text
    for t in sorted((t for t in diagnosis_terms(disease) if len(t) >= 2), key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(t) + r"\b", placeholder, text, flags=re.IGNORECASE)
    return text


_GENERIC_GUIDELINE = "https://pubmed.ncbi.nlm.nih.gov/?term=clinical+practice+guideline"


def _alnum(s: str) -> str:
    # URL-decode first so %20/%2B-encoded names collapse to the same form as the
    # plain diagnosis (otherwise "myocardial%20infarction" -> "...20..." misses).
    return re.sub(r"[^a-z0-9]", "", unquote(s or "").lower())


def redact_citation(url: str, disease: dict) -> str:
    """Neutralise a citation URL that would spell out the diagnosis in its query
    or slug (e.g. ``?term=Myocardial+Infarction+guideline`` or ``/stemi-guideline``).

    The word-boundary text redactor can't see these because URL encoding joins
    words with ``+``/``%20``/``-``. We compare on a decoded, alphanumeric-only
    form and, on a hit, swap in a diagnosis-free literature search. Terms <3 chars
    are ignored to avoid matching incidental substrings in unrelated URLs.
    """
    if not url:
        return url
    norm = _alnum(url)
    for t in diagnosis_terms(disease):
        tn = _alnum(t)
        if len(tn) >= 3 and tn in norm:
            return _GENERIC_GUIDELINE
    return url

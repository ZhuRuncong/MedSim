"""Small shared helpers."""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

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

"""symptom_generator — POST /api/v1/symptoms.

Sample which of a disease's symptoms are present in *this* patient, weighted by
the epidemiological frequency table (common/occasional/rare), per PLAN §3.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# Probability that a symptom of a given frequency is present in a given patient.
_FREQUENCY_P = {"common": 0.9, "occasional": 0.55, "rare": 0.2}


def generate_symptoms(disease: dict, rng: Optional[np.random.Generator] = None) -> dict:
    """Return the present symptoms + chief complaint for one instantiation.

    ``rng`` lets callers seed by case id for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    present: List[str] = []
    for s in disease.get("symptoms", []):
        p = _FREQUENCY_P.get(s.get("frequency", "occasional"), 0.5)
        # numpy weighted choice between present/absent (PLAN: numpy.random.choice)
        if rng.choice([True, False], p=[p, 1 - p]):
            present.append(s["name"])

    # Guarantee a non-empty, informative presentation: fall back to the
    # "common" symptoms if the sampling happened to drop everything.
    if not present:
        present = [s["name"] for s in disease.get("symptoms", [])
                   if s.get("frequency") == "common"]

    citation = disease.get("guideline", {}).get("url")
    return {
        "disease_id": disease.get("id"),
        "chief_complaint": disease.get("chief_complaint", ""),
        "present_symptoms": present,
        "citations": [c for c in [citation] if c],
    }

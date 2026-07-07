"""emergency_surgery — POST /api/v1/surgery.

Validate whether a requested procedure is indicated by the disease's guidelines,
and return an outcome probability plus its point impact. PLAN §3.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import data_loader


def evaluate_surgery(
    procedure: str,
    disease: dict,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    indicated_list = disease.get("indicated_surgeries", [])
    indicated = procedure in indicated_list

    if indicated:
        success_probability = 0.9
        success = bool(rng.random() < success_probability)
        outcome = (
            "Procedure indicated and performed successfully; the patient stabilises."
            if success else
            "Procedure was indicated but complicated by a peri-operative event; the patient recovers with support."
        )
        rationale = f"{procedure} is guideline-indicated for {disease.get('name')}."
    else:
        success_probability = 0.15
        success = False
        if indicated_list:
            outcome = (f"{procedure} is not indicated for this presentation. "
                       f"An unnecessary operation exposes the patient to avoidable surgical risk.")
        else:
            outcome = (f"This condition is managed medically. {procedure} is not indicated "
                       f"and would be harmful.")
        rationale = (f"{procedure} is not among the indicated procedures "
                     f"({', '.join(indicated_list) or 'none'}) for {disease.get('name')}.")

    citation = disease.get("guideline", {}).get("url")
    return {
        "procedure": procedure,
        "disease_id": disease.get("id"),
        "indicated": indicated,
        "success": success,
        "success_probability": success_probability,
        "outcome": outcome,
        "rationale": rationale,
        "indicated_procedures": indicated_list,
        "citations": [c for c in [citation] if c],
    }

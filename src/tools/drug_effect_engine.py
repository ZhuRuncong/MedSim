"""drug_effect_engine — POST /api/v1/drug.

Check allergy + drug-interaction matrix and grade the drug against the disease's
guideline first-line / contraindicated lists. Returns an effect score, a
contraindication flag and citations. PLAN §3.
"""
from __future__ import annotations

from typing import List, Optional

from .. import data_loader

# Illustrative pharmacodynamic effect score (distinct from the game rubric).
_EFFECT = {
    "first_line": 2,
    "reasonable": 1,
    "neutral": 0,
    "contraindicated": -2,
    "allergen": -3,
}

_PUBMED = "https://pubmed.ncbi.nlm.nih.gov/?term="


def evaluate_drug(
    drug: str,
    disease: dict,
    allergies: Optional[List[str]] = None,
    current_drugs: Optional[List[str]] = None,
) -> dict:
    allergies = allergies or []
    current_drugs = current_drugs or []

    meta = data_loader.drug_info(drug) or {}
    first_line = disease.get("first_line_drugs", [])
    reasonable = disease.get("reasonable_drugs", [])
    contraindicated = disease.get("contraindicated_drugs", [])

    allergy = data_loader.allergy_conflict(drug, allergies)
    interactions = data_loader.interactions_for(drug, current_drugs)

    # Category precedence: safety concerns override efficacy.
    if allergy and allergy["type"] == "allergen":
        category = "allergen"
    elif drug in contraindicated:
        category = "contraindicated"
    elif drug in first_line:
        category = "first_line"
    elif drug in reasonable:
        category = "reasonable"
    else:
        category = "neutral"

    contraindication_flag = category in ("allergen", "contraindicated")
    has_major_interaction = any(i.get("severity") == "major" for i in interactions)

    # Rationales must never name the diagnosis — the student is still working it
    # out. Describe the drug's role relative to "this presentation" instead.
    rationale_map = {
        "first_line": f"{drug} is a guideline first-line agent for this presentation.",
        "reasonable": f"{drug} is a reasonable adjunct or alternative here.",
        "neutral": f"{drug} is not indicated for this presentation and offers no benefit here.",
        "contraindicated": f"{drug} is contraindicated for this patient.",
        "allergen": (allergy or {}).get("description", f"{drug} conflicts with a patient allergy."),
    }

    citations = []
    g = disease.get("guideline", {}).get("url")
    if g:
        citations.append(g)
    # Drug-only query — including the diagnosis in the URL would leak the answer.
    citations.append(_PUBMED + drug.replace(" ", "+") + "+treatment+guideline")

    return {
        "drug": drug,
        "drug_class": meta.get("class"),
        "category": category,
        "effect_score": _EFFECT[category],
        "contraindication_flag": contraindication_flag,
        "allergy": allergy,
        "interactions": interactions,
        "has_major_interaction": has_major_interaction,
        "efficacy": "guideline first-line" if category == "first_line" else category,
        "rationale": rationale_map[category],
        "citations": citations,
    }

"""Clinical reference data + the runtime disease registry.

MedSim ships NO hard-coded disease cases: every playable case is authored at
runtime by the AI generator (or, in tests, injected as a fixture). This module
holds the *reference vocabulary* the simulator is built on — drugs, drug
interactions, lab reference ranges, qualitative tests — loaded from ``data/``,
plus an in-memory disease **registry** that ``register_disease`` populates.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from typing import Dict, List, Optional

from . import config

# Qualitative / imaging tests: they produce a textual "finding" rather than a
# numeric panel. Handled specially by the lab simulator.
QUALITATIVE_TESTS = [
    "Chest X-ray",
    "ECG",
    "Abdominal Ultrasound",
    "CT Abdomen/Pelvis",
    "Transvaginal Ultrasound",
    "Pelvic Ultrasound",
    "RSV Antigen",
    "Rapid Strep Test",
    "Blood Culture",
    "Sputum Culture",
    "Urine Culture",
    "Wound Culture",
    "Skin Swab",
    "Urinalysis",
    "Serum Ketones",
    "Blood Type & Screen",
    "Urine Protein/Creatinine Ratio",
    "Pulse Oximetry",
]


# --------------------------------------------------------------------------- #
# Reference-vocabulary file loaders (cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _drug_db_raw() -> dict:
    with open(config.DRUG_INTERACTIONS_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _lab_rows() -> List[dict]:
    with open(config.LAB_REFS_CSV, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# Disease registry — starts EMPTY; populated only by register_disease()
# (AI-generated cases at runtime; fixture cases under test).
# --------------------------------------------------------------------------- #
_REGISTRY: "Dict[str, dict]" = {}
_ORDER: "List[str]" = []


def diseases() -> List[dict]:
    return [_REGISTRY[i] for i in _ORDER]


def get_disease(disease_id: str) -> Optional[dict]:
    return _REGISTRY.get(disease_id)


def diseases_for_specialties(specialties: List[str]) -> List[dict]:
    """Registered diseases whose specialties intersect the requested ones."""
    wanted = set(specialties)
    if not wanted:
        return diseases()
    return [d for d in diseases() if wanted.intersection(d.get("specialties", []))]


# --------------------------------------------------------------------------- #
# Drug helpers
# --------------------------------------------------------------------------- #
def drug_db() -> dict:
    return _drug_db_raw()


def drug_info(name: str) -> Optional[dict]:
    return _drug_db_raw().get("drugs", {}).get(name)


def all_drug_names() -> List[str]:
    # Exclude non-drug diagnostic entries.
    return sorted(
        n for n, meta in _drug_db_raw().get("drugs", {}).items()
        if meta.get("class") != "diagnostic"
    )


@lru_cache(maxsize=1)
def _interaction_matrix() -> Dict[str, Dict[str, dict]]:
    """Symmetric drug-interaction lookup built from the (directional) JSON."""
    raw = _drug_db_raw().get("interactions", {})
    matrix: Dict[str, Dict[str, dict]] = {}
    for a, partners in raw.items():
        for b, info in partners.items():
            matrix.setdefault(a, {})[b] = info
            matrix.setdefault(b, {})[a] = info
    return matrix


def interactions_for(drug: str, others: List[str]) -> List[dict]:
    """Return interaction records between ``drug`` and each of ``others``."""
    matrix = _interaction_matrix()
    hits = []
    for other in others:
        if other == drug:
            continue
        info = matrix.get(drug, {}).get(other)
        if info:
            hits.append({"with": other, **info})
    return hits


def allergy_conflict(drug: str, allergies: List[str]) -> Optional[dict]:
    """If ``drug`` conflicts with one of the patient's allergies, describe it."""
    if not allergies:
        return None
    db = _drug_db_raw()
    classes = db.get("allergy_classes", {})
    cross = db.get("cross_reactivity", {})
    meta = db.get("drugs", {}).get(drug, {})
    drug_allergy_class = meta.get("allergy_class")
    for allergy in allergies:
        a = allergy.strip().lower()
        if not a or a == "none":
            continue
        members = classes.get(a, [])
        # Direct allergy: drug belongs to the allergen class.
        if drug in members or drug_allergy_class == a:
            return {"type": "allergen", "allergy": a, "severity": "contraindicated",
                    "description": f"Patient reports a {a} allergy; {drug} is in that class."}
        # Cross-reactivity (softer caution) — clinically symmetric, so check
        # both directions (penicillin-allergic → cephalosporin drug AND
        # cephalosporin-allergic → penicillin drug).
        xr = (cross.get(a, {}).get(drug_allergy_class)
              or cross.get(drug_allergy_class, {}).get(a))
        if xr:
            return {"type": "cross_reactivity", "allergy": a,
                    "severity": xr.get("severity", "caution"),
                    "description": xr.get("description", "")}
    return None


# --------------------------------------------------------------------------- #
# Lab helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _lab_index() -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = {}
    for row in _lab_rows():
        idx.setdefault(row["test"], []).append({
            "component": row["component"],
            "low": float(row["low"]),
            "high": float(row["high"]),
            "unit": row["unit"],
        })
    return idx


def numeric_panels() -> List[str]:
    return list(_lab_index().keys())


def components_for_test(test: str) -> List[dict]:
    return _lab_index().get(test, [])


def orderable_tests() -> List[str]:
    """Every test the student can order (numeric panels + qualitative)."""
    return sorted(set(numeric_panels()) | set(QUALITATIVE_TESTS))


@lru_cache(maxsize=1)
def _component_to_panel() -> Dict[str, str]:
    return {c["component"]: panel
            for panel in numeric_panels()
            for c in components_for_test(panel)}


def panel_for_component(component: str) -> Optional[str]:
    """Which orderable panel a numeric lab component belongs to (e.g. WBC→CBC)."""
    return _component_to_panel().get(component)


def all_components() -> set:
    return set(_component_to_panel().keys())


# --------------------------------------------------------------------------- #
# Runtime-registered (AI-generated) cases
# --------------------------------------------------------------------------- #
def register_disease(disease: dict) -> None:
    """Add a disease dict to the in-memory registry so get_disease / case
    selection can see it. Idempotent by id."""
    if disease["id"] not in _REGISTRY:
        _ORDER.append(disease["id"])
    _REGISTRY[disease["id"]] = disease


def persist_generated(disease: dict) -> str:
    """Write a generated case to the cache dir so it survives a restart."""
    import json

    config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GENERATED_DIR / f"{disease['id']}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(disease, fh, indent=2)
    return str(path)


def load_generated_cases() -> int:
    """Register every cached generated case. Returns how many were loaded."""
    import json

    if not config.GENERATED_DIR.exists():
        return 0
    n = 0
    for path in sorted(config.GENERATED_DIR.glob("*.json")):
        if path.name == "drugs.json":
            continue  # not a case
        try:
            with open(path, "r", encoding="utf-8") as fh:
                register_disease(json.load(fh))
            n += 1
        except Exception:
            continue
    return n


# --------------------------------------------------------------------------- #
# Self-extending drug vocabulary (AI-authored drugs)
# --------------------------------------------------------------------------- #
def known_allergy_families() -> set:
    """Allergy classes the engine understands (a new drug must map to one or 'none')."""
    return set(_drug_db_raw().get("allergy_classes", {}).keys())


def register_drug(name: str, meta: dict) -> None:
    """Add a drug to the in-memory catalog so tools/agents/UI can use it.
    Never overwrites a curated drug. Idempotent.

    ``meta`` may carry an ``interactions`` sub-dict
    (``{partner: {severity, description}}``); it is merged into the interaction
    matrix (which is symmetric) and kept in ``meta`` for persistence."""
    db = _drug_db_raw()
    drugs = db.setdefault("drugs", {})
    if name in drugs and not drugs[name].get("generated"):
        return  # protect curated entries
    drugs[name] = meta
    ac = meta.get("allergy_class")
    if ac and ac != "none":  # keep the allergy-class membership lists consistent
        members = db.setdefault("allergy_classes", {}).setdefault(ac, [])
        if name not in members:
            members.append(name)

    interactions = meta.get("interactions") or {}
    if interactions:
        table = db.setdefault("interactions", {}).setdefault(name, {})
        for partner, info in interactions.items():
            table[partner] = info
        _interaction_matrix.cache_clear()  # rebuild symmetric matrix on next read


_GEN_DRUGS_FILE = "drugs.json"


def persist_generated_drug(name: str, meta: dict) -> None:
    import json

    config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GENERATED_DIR / _GEN_DRUGS_FILE
    data = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    data[name] = meta
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_generated_drugs() -> int:
    import json

    path = config.GENERATED_DIR / _GEN_DRUGS_FILE
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return 0
    for name, meta in data.items():
        register_drug(name, meta)
    return len(data)


def load_generated() -> dict:
    """Load cached generated drugs (first, so cases can reference them) + cases."""
    return {"drugs": load_generated_drugs(), "cases": load_generated_cases()}

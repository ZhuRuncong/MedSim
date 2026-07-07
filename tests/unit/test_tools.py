"""Unit tests for the FastMCP tool functions (PLAN §8)."""
import numpy as np
import pytest

from src import data_loader
from src.tools import (evaluate_drug, evaluate_surgery, generate_symptoms,
                       lookup, simulate_labs)

PNEUMONIA = "C0032285"
MI = "C0155626"
APPENDICITIS = "C0003615"
DKA = "C0011880"


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def d(disease_id):
    return data_loader.get_disease(disease_id)


# --- symptom_generator ---------------------------------------------------- #
def test_generate_symptoms_shape(rng):
    out = generate_symptoms(d(PNEUMONIA), rng)
    assert out["disease_id"] == PNEUMONIA
    assert out["chief_complaint"]
    assert isinstance(out["present_symptoms"], list) and out["present_symptoms"]
    assert out["citations"]  # guideline URL present


def test_generate_symptoms_never_empty():
    # Even a hostile RNG must not yield an empty presentation.
    for seed in range(20):
        out = generate_symptoms(d(PNEUMONIA), np.random.default_rng(seed))
        assert out["present_symptoms"]


# --- lab_simulator -------------------------------------------------------- #
def test_labs_panel_applies_deviation(rng):
    out = simulate_labs(["CBC"], d(PNEUMONIA), rng=rng)
    cbc = out["results"]["CBC"]
    assert cbc["type"] == "panel"
    assert cbc["components"]["WBC"]["value"] == 15.4
    assert cbc["components"]["WBC"]["flag"] == "H"


def test_labs_qualitative_finding(rng):
    out = simulate_labs(["Chest X-ray"], d(PNEUMONIA), rng=rng)
    cxr = out["results"]["Chest X-ray"]
    assert cxr["type"] == "qualitative"
    assert "consolidation" in cxr["finding"].lower()


def test_labs_normal_panel_when_no_deviation(rng):
    # An unrelated panel on pneumonia should read essentially normal.
    out = simulate_labs(["Lipid Panel"], d(PNEUMONIA), rng=rng)
    comps = out["results"]["Lipid Panel"]["components"]
    assert all(c["flag"] == "N" for c in comps.values())


def test_labs_unavailable_test(rng):
    out = simulate_labs(["Nonexistent Test"], d(PNEUMONIA), rng=rng)
    assert out["results"]["Nonexistent Test"]["type"] == "unavailable"


def test_labs_direction_low_forces_L_flag(rng):
    # A 'low' directional deviation must be flagged L even after rounding.
    synthetic = {"id": "X", "vitals": {}, "guideline": {},
                 "lab_deviations": {"Total Bilirubin": {"direction": "low"}}}
    out = simulate_labs(["LFT"], synthetic, rng=rng)
    assert out["results"]["LFT"]["components"]["Total Bilirubin"]["flag"] == "L"


def test_labs_direction_high_forces_H_flag(rng):
    synthetic = {"id": "X", "vitals": {}, "guideline": {},
                 "lab_deviations": {"Potassium": {"direction": "high"}}}
    out = simulate_labs(["BMP"], synthetic, rng=rng)
    assert out["results"]["BMP"]["components"]["Potassium"]["flag"] == "H"


def test_allergy_cross_reactivity_is_symmetric():
    from src import data_loader
    # penicillin-allergic → cephalosporin drug (the direction present in data)
    assert data_loader.allergy_conflict("Ceftriaxone", ["penicillin"]) is not None
    # cephalosporin-allergic → penicillin drug (the reverse must also warn)
    rev = data_loader.allergy_conflict("Amoxicillin", ["cephalosporin"])
    assert rev is not None and rev["type"] == "cross_reactivity"


# --- drug_effect_engine --------------------------------------------------- #
def test_drug_first_line():
    ev = evaluate_drug("Amoxicillin", d(PNEUMONIA), allergies=[], current_drugs=[])
    assert ev["category"] == "first_line"
    assert ev["effect_score"] > 0
    assert ev["contraindication_flag"] is False
    assert ev["citations"]


def test_drug_allergen_overrides_efficacy():
    # Amoxicillin is first-line for CAP but a penicillin allergy makes it an allergen.
    ev = evaluate_drug("Amoxicillin", d(PNEUMONIA), allergies=["penicillin"], current_drugs=[])
    assert ev["category"] == "allergen"
    assert ev["contraindication_flag"] is True


def test_drug_contraindicated():
    ev = evaluate_drug("Ibuprofen", d(MI), allergies=[], current_drugs=[])
    assert ev["category"] == "contraindicated"
    assert ev["contraindication_flag"] is True


def test_drug_major_interaction():
    ev = evaluate_drug("Nitroglycerin", d(MI), allergies=[], current_drugs=["Sildenafil"])
    assert ev["has_major_interaction"] is True
    assert any(i["with"] == "Sildenafil" for i in ev["interactions"])


def test_drug_neutral():
    ev = evaluate_drug("Metformin", d(PNEUMONIA), allergies=[], current_drugs=[])
    assert ev["category"] == "neutral"
    assert ev["effect_score"] == 0


# --- knowledge_lookup ----------------------------------------------------- #
def test_lookup_local_guideline():
    out = lookup("first-line treatment for community-acquired pneumonia?")
    assert out["source"].startswith("local")
    assert out["citations"]
    assert "amoxicillin" in out["answer"].lower()


def test_lookup_unknown_returns_no_source():
    out = lookup("what is the capital of France")
    assert out["source"] == "none"
    assert out["citations"] == []


# --- emergency_surgery ---------------------------------------------------- #
def test_surgery_indicated(rng):
    out = evaluate_surgery("Appendectomy", d(APPENDICITIS), rng)
    assert out["indicated"] is True
    assert out["success_probability"] > 0.5
    assert out["citations"]


def test_surgery_not_indicated(rng):
    out = evaluate_surgery("Appendectomy", d(PNEUMONIA), rng)
    assert out["indicated"] is False

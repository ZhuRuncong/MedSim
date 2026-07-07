"""Tests for the AI case generator + generate-then-verify harness.

All tests use an injected MockClient, so they exercise the full pipeline
(schema validation, integrity checks, verification voting, retries, registration
and playability) with zero network access.
"""
import copy

import pytest

from src import config, data_loader, engine, generator
from src.llm import LLMUnavailable, MockClient, get_client

PASS = {"status": "pass", "confidence": 0.9, "issues": []}
REJECT = {"status": "reject", "confidence": 0.2, "issues": ["contraindicated drug is wrong"]}


def good_case():
    """A structurally + medically valid case an LLM might emit (raw JSON)."""
    return {
        "name": "Acute Pericarditis",
        "aliases": ["pericarditis"],
        "specialties": ["Emergency"],
        "difficulty": 2,
        "chief_complaint": "Sharp chest pain relieved by leaning forward",
        "demographics": {"age_range": [25, 45], "sex": "any"},
        "vitals": {"HR": 98, "SBP": 124, "DBP": 78, "RR": 18, "Temp": 37.6, "SpO2": 98},
        "symptoms": [
            {"name": "pleuritic chest pain", "frequency": "common"},
            {"name": "pain relieved by sitting forward", "frequency": "common"},
            {"name": "low-grade fever", "frequency": "occasional"},
        ],
        "history": {"onset": "It started two days ago.",
                    "chest pain": "Sharp, worse lying flat, better leaning forward."},
        "exam_findings": {"Auscultate heart": "A three-component pericardial friction rub."},
        "appropriate_tests": ["ECG", "Troponin", "CRP", "CBC"],
        "appropriate_exams": ["Auscultate heart"],
        "lab_deviations": {"CRP": {"value": 45},
                           "ECG": {"finding": "Diffuse ST elevation with PR depression."}},
        "first_line_drugs": ["Ibuprofen"],
        "reasonable_drugs": ["Acetaminophen"],
        "contraindicated_drugs": [],
        "indicated_surgeries": [],
        "differentials": ["Acute Myocardial Infarction", "Pulmonary Embolism"],
        "teaching": "NSAIDs (plus colchicine) are first-line for acute pericarditis.",
        "guideline": {"name": "ESC Pericardial Diseases 2015",
                      "url": "https://doi.org/10.1093/eurheartj/ehv318"},
    }


# --- validation ----------------------------------------------------------- #
def test_validate_accepts_good_case():
    assert generator.validate_disease(good_case()) == []


def test_validate_flags_unknown_drug():
    c = good_case()
    c["first_line_drugs"] = ["Foobarcillin"]
    problems = generator.validate_disease(c)
    assert any("unknown drug" in p for p in problems)


def test_validate_flags_first_line_and_contraindicated_overlap():
    c = good_case()
    c["contraindicated_drugs"] = ["Ibuprofen"]  # also first-line
    problems = generator.validate_disease(c)
    assert any("both first-line AND contraindicated" in p for p in problems)


def test_validate_flags_unorderable_test_and_bad_exam():
    c = good_case()
    c["appropriate_tests"].append("Crystal Ball Scan")
    c["exam_findings"]["Read minds"] = "nothing"
    problems = generator.validate_disease(c)
    assert any("not orderable" in p for p in problems)
    assert any("PHYSICAL_EXAMS" in p for p in problems)


# --- verification voting -------------------------------------------------- #
def test_verify_majority_rejects_split_vote():
    client = MockClient([PASS, REJECT])
    v = generator.verify_case(good_case(), client, n_verifiers=2, min_confidence=0.7)
    assert v["accepted"] is False


def test_verify_accepts_unanimous_high_confidence():
    client = MockClient([PASS, dict(PASS)])
    v = generator.verify_case(good_case(), client, n_verifiers=2, min_confidence=0.7)
    assert v["accepted"] is True
    assert v["confidence"] >= 0.7


# --- generate loop -------------------------------------------------------- #
def test_generate_success_registers_and_tags_provenance():
    client = MockClient([good_case(), PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    assert d["id"].startswith("GEN-")
    assert d["generated"] is True
    assert d["provenance"]["source"] == "ai-generated"
    assert d["provenance"]["reviewed_by_human"] is False
    assert d["provenance"]["model"] == "mock"


def test_generate_retries_on_invalid_then_succeeds():
    bad = good_case()
    bad["first_line_drugs"] = ["Foobarcillin"]      # fails integrity validation
    client = MockClient([bad, good_case(), PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    assert d["name"] == "Acute Pericarditis"
    # 1 failed gen (no verify) + 1 good gen + 1 verify = 3 calls
    assert len(client.calls) == 3


def test_generate_repairs_gates_and_clamps_severity():
    case = good_case()
    case["severity"] = 7                                    # out of range -> clamped to 3
    case["safety_gates"] = [
        {"type": "test_before_drug", "test": "CBC", "drug": "Ibuprofen",
         "rationale": "ok", "penalty": 10},                 # valid (both in vocab)
        {"type": "test_before_drug", "test": "CBC", "drug": "Foobarcillin"},  # unknown drug -> dropped
        {"type": "test_before_test", "first": "Nope Test", "then": "ECG"},    # unknown test -> dropped
    ]
    d = generator.generate_case("Emergency", client=MockClient([case, PASS]),
                                n_verifiers=1, register=False)
    assert d["severity"] == 3
    assert len(d["safety_gates"]) == 1
    assert d["safety_gates"][0]["drug"] == "Ibuprofen"
    assert "CBC" in d["appropriate_tests"]                  # gate prerequisite made appropriate


def test_generate_filters_out_of_vocab_specialty():
    # Model tags an unlisted specialty; it should be auto-repaired, not retried.
    case = good_case()
    case["specialties"] = ["Emergency", "Gastroenterology"]
    client = MockClient([case, PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    assert "Gastroenterology" not in d["specialties"]
    assert "Emergency" in d["specialties"]
    assert len(client.calls) == 2  # no wasted retry attempt


def test_generate_repairs_unknown_labs_and_drugs():
    # Model proposes an unknown lab (Hematocrit) and a mix of known/unknown drugs.
    case = good_case()
    case["lab_deviations"] = {"Hematocrit": {"value": 30}, "CRP": {"value": 45}}
    case["first_line_drugs"] = ["Colchicine", "Ibuprofen"]  # Colchicine not in DB
    client = MockClient([case, PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    assert "Hematocrit" not in d["lab_deviations"] and "CRP" in d["lab_deviations"]
    assert d["first_line_drugs"] == ["Ibuprofen"]           # kept the valid one
    assert generator.validate_disease(d) == []              # fully valid after repair
    assert len(client.calls) == 2                            # no wasted retry


def test_generate_retries_when_verification_rejects():
    client = MockClient([good_case(), REJECT, good_case(), PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    assert d["provenance"]["attempts"] == 2


def test_generate_raises_after_max_attempts():
    bad = good_case()
    bad["appropriate_tests"] = ["Nope Test"]
    client = MockClient([copy.deepcopy(bad) for _ in range(3)])
    with pytest.raises(generator.CaseGenerationError):
        generator.generate_case("Emergency", client=client, n_verifiers=1,
                                max_attempts=3, register=False)


def test_generate_raises_without_provider():
    # No API key configured in the test environment.
    assert get_client() is None
    with pytest.raises(LLMUnavailable):
        generator.generate_case("Emergency", client=None)


# --- end-to-end playability ---------------------------------------------- #
def test_generated_case_is_playable():
    client = MockClient([good_case(), PASS])
    disease = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    data_loader.register_disease(disease)  # in-memory only (no disk)

    st = engine.create_case(["Emergency"], disease_id=disease["id"])
    st.allergies = []
    engine.perform_action(st, {"type": "order_test", "payload": "ECG"})       # appropriate
    engine.perform_action(st, {"type": "prescribe_drug", "payload": "Ibuprofen"})  # first-line
    engine.perform_action(st, {"type": "submit_diagnosis", "payload": "pericarditis"})

    actions = [e.action for e in st.score_log]
    assert "order_appropriate_test" in actions
    assert "prescribe_guideline_first_line" in actions
    assert "correct_diagnosis" in actions
    assert st.status == "complete"


# --- persistence ---------------------------------------------------------- #
def test_generate_introduces_new_drug():
    # The model authors a case whose first-line drug isn't in the base formulary.
    case = good_case()
    case["first_line_drugs"] = ["Insulin Glargine"]
    case["new_drugs"] = [{"name": "Insulin Glargine", "drug_class": "long-acting insulin",
                          "allergy_class": "none", "route": "SC",
                          "monograph": "Basal insulin analogue."}]
    client = MockClient([case, PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    assert "Insulin Glargine" in d["first_line_drugs"]
    assert data_loader.drug_info("Insulin Glargine")["generated"] is True
    assert "Insulin Glargine" in data_loader.all_drug_names()
    assert d["provenance"]["new_drugs"] == ["Insulin Glargine"]


def test_generated_drug_carries_interactions():
    case = good_case()
    case["first_line_drugs"] = ["Novacoagulant"]
    case["new_drugs"] = [{
        "name": "Novacoagulant", "drug_class": "novel anticoagulant", "allergy_class": "none",
        "route": "PO", "monograph": "Investigational DOAC.",
        "interactions": [
            {"with": "Aspirin", "severity": "major", "description": "additive bleeding risk"},
            {"with": "Nonexistent Drug", "severity": "major", "description": "bogus"},   # dropped
            {"with": "Ibuprofen", "severity": "banana", "description": "bad severity"},  # dropped
        ]}]
    generator.generate_case("Emergency", client=MockClient([case, PASS]),
                            n_verifiers=1, register=False)
    # Registered + symmetric in the interaction matrix
    fwd = data_loader.interactions_for("Novacoagulant", ["Aspirin"])
    rev = data_loader.interactions_for("Aspirin", ["Novacoagulant"])
    assert fwd and fwd[0]["severity"] == "major"
    assert rev and rev[0]["with"] == "Novacoagulant"
    # Invalid interactions were dropped, not fatal
    assert data_loader.interactions_for("Novacoagulant", ["Ibuprofen"]) == []
    assert data_loader.drug_info("Novacoagulant")["interactions"] == {
        "Aspirin": {"severity": "major", "description": "additive bleeding risk"}}


def test_generated_drug_interaction_penalised_in_engine():
    case = good_case()
    case["first_line_drugs"] = ["Bleedomab"]
    case["new_drugs"] = [{"name": "Bleedomab", "drug_class": "anticoagulant", "allergy_class": "none",
                          "route": "IV", "monograph": "x",
                          "interactions": [{"with": "Aspirin", "severity": "major",
                                            "description": "bleeding"}]}]
    d = generator.generate_case("Emergency", client=MockClient([case, PASS]),
                                n_verifiers=1, register=False)
    data_loader.register_disease(d)
    st = engine.create_case(["Emergency"], disease_id=d["id"]); st.allergies = []
    engine.perform_action(st, {"type": "prescribe_drug", "payload": "Aspirin"})
    msgs = engine.perform_action(st, {"type": "prescribe_drug", "payload": "Bleedomab"})
    assert any("major" in m.text.lower() and "aspirin" in m.text.lower() for m in msgs)
    assert any(e.action == "major_drug_interaction" for e in st.score_log)


def test_generated_drug_bad_allergy_class_is_rejected():
    case = good_case()
    case["first_line_drugs"] = ["Mystery Med"]
    case["new_drugs"] = [{"name": "Mystery Med", "drug_class": "x",
                          "allergy_class": "banana"}]  # not a known family
    good = good_case()  # valid retry
    client = MockClient([case, good, PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    assert d["name"] == "Acute Pericarditis"      # recovered on the retry
    assert data_loader.drug_info("Mystery Med") is None  # never registered


def test_generated_drug_respects_declared_allergy_family():
    # A new drug that declares a penicillin allergy family must trip the allergy engine.
    case = good_case()
    case["first_line_drugs"] = ["Novacillin"]
    case["new_drugs"] = [{"name": "Novacillin", "drug_class": "novel penicillin",
                          "allergy_class": "penicillin", "route": "IV",
                          "monograph": "Investigational beta-lactam."}]
    client = MockClient([case, PASS])
    generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    conflict = data_loader.allergy_conflict("Novacillin", ["penicillin"])
    assert conflict is not None and conflict["type"] == "allergen"


def test_persist_and_load_generated_drug_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GENERATED_DIR", tmp_path)
    data_loader.persist_generated_drug("ZZDrug", {"class": "test", "allergy_class": "none",
                                                  "route": "PO", "monograph": "", "generated": True})
    assert (tmp_path / "drugs.json").exists()
    assert data_loader.load_generated_drugs() >= 1
    assert data_loader.drug_info("ZZDrug") is not None


def test_engine_create_generated_case(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GENERATED_DIR", tmp_path)  # keep repo clean
    # create_generated_case uses the configured default of GEN_VERIFIERS (2).
    client = MockClient([good_case(), PASS, dict(PASS)])
    st = engine.create_generated_case("Emergency", difficulty=2, client=client)
    assert st.status == "active"
    assert data_loader.get_disease(st.disease_id).get("generated") is True
    # the case is immediately playable
    engine.perform_action(st, {"type": "submit_diagnosis", "payload": "pericarditis"})
    assert st.status == "complete"


def test_persist_and_load_generated_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GENERATED_DIR", tmp_path)
    client = MockClient([good_case(), PASS])
    d = generator.generate_case("Emergency", client=client, n_verifiers=1, register=False)
    path = data_loader.persist_generated(d)
    assert (tmp_path / f"{d['id']}.json").exists()
    loaded = data_loader.load_generated_cases()
    assert loaded >= 1
    assert data_loader.get_disease(d["id"]) is not None

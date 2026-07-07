"""Offline tests for the real-world grounding transforms (no network)."""
from src import grounding


# --- ATC → allergy family (safety-critical) ------------------------------- #
def test_atc_to_allergy_family():
    assert grounding.atc_to_allergy_family("J01CA04") == "penicillin"   # amoxicillin
    assert grounding.atc_to_allergy_family("J01DD04") == "cephalosporin"  # ceftriaxone
    assert grounding.atc_to_allergy_family("J01FA10") == "macrolide"    # azithromycin
    assert grounding.atc_to_allergy_family("M01AE01") == "nsaid"        # ibuprofen
    assert grounding.atc_to_allergy_family("A10AB01") is None           # insulin — no family
    assert grounding.atc_to_allergy_family(None) is None


# --- RxClass parsing ------------------------------------------------------ #
def test_parse_rxclass_extracts_epc_and_atc():
    data = {"rxclassDrugInfoList": {"rxclassDrugInfo": [
        {"rxclassMinConceptItem": {"classId": "N0000175503", "className": "Penicillin-class Antibacterial", "classType": "EPC"}},
        {"rxclassMinConceptItem": {"classId": "J01CA", "className": "Penicillins with extended spectrum", "classType": "ATC1-4"}},
        {"rxclassMinConceptItem": {"classId": "J01", "className": "Antibacterials", "classType": "ATC1-4"}},
    ]}}
    out = grounding.parse_rxclass(data)
    assert out["class"] == "Penicillin-class Antibacterial"
    assert out["atc"] == "J01CA"   # most specific ATC kept


# --- openFDA label parsing ------------------------------------------------ #
def test_parse_openfda_label():
    data = {"results": [{
        "indications_and_usage": ["1 INDICATIONS AND USAGE Amoxicillin is indicated for infections. Extra text."],
        "contraindications": ["4 CONTRAINDICATIONS History of serious hypersensitivity to penicillins."],
        "boxed_warning": [],
        "openfda": {"spl_set_id": ["abc-123"], "rxcui": ["723"], "pharm_class_epc": ["Penicillin-class Antibacterial"]},
    }]}
    out = grounding.parse_openfda_label(data)
    assert out["label_url"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc-123"
    assert "indicated for infections" in out["monograph"]
    assert "hypersensitivity" in out["contraindications"]
    assert out["rxcui"] == "723"
    assert "boxed_warning" not in out


def test_parse_openfda_label_empty():
    assert grounding.parse_openfda_label({"results": []}) == {}


# --- DDInter parsing ------------------------------------------------------ #
def test_ddinter_name_map_and_parse():
    formulary = ["Warfarin", "Ibuprofen", "Trimethoprim-Sulfamethoxazole", "Normal Saline"]
    nm = grounding.ddinter_name_map(formulary)
    assert nm["warfarin"] == "Warfarin"
    assert nm["sulfamethoxazole"] == "Trimethoprim-Sulfamethoxazole"  # alias mapping
    rows = [
        {"Drug_A": "Warfarin", "Drug_B": "Ibuprofen", "Level": "Major"},
        {"Drug_A": "Sulfamethoxazole", "Drug_B": "Warfarin", "Level": "Moderate"},
        {"Drug_A": "Warfarin", "Drug_B": "SomethingWeDontStock", "Level": "Major"},  # dropped
    ]
    inter = grounding.parse_ddinter(rows, nm)
    assert inter["Warfarin"]["Ibuprofen"]["severity"] == "major"
    assert inter["Trimethoprim-Sulfamethoxazole"]["Warfarin"]["severity"] == "moderate"
    assert "SomethingWeDontStock" not in inter.get("Warfarin", {})


# --- ONC overrides + merge ------------------------------------------------ #
def _db():
    return {"drugs": {
        "Nitroglycerin": {"allergy_class": "none", "class": "nitrate"},
        "Sildenafil": {"allergy_class": "none", "class": "PDE5 inhibitor"},
        "Warfarin": {"allergy_class": "none", "class": "anticoagulant"},
        "Ibuprofen": {"allergy_class": "nsaid", "class": "NSAID"},
        "Naproxen": {"allergy_class": "nsaid", "class": "NSAID"},
    }}


def test_onc_override_expands_families_and_forces_major():
    db = _db()
    inter = grounding.apply_onc_overrides({}, db)
    assert inter["Nitroglycerin"]["Sildenafil"]["severity"] == "major"
    # family:nsaid + Warfarin -> every nsaid drug paired with Warfarin, forced major
    assert inter["Ibuprofen"]["Warfarin"]["severity"] == "major"
    assert inter["Naproxen"]["Warfarin"]["severity"] == "major"
    assert inter["Ibuprofen"]["Warfarin"]["source"] == "onc"


def test_merge_precedence_onc_over_curated_over_ddinter():
    db = _db()
    curated = {"Nitroglycerin": {"Sildenafil": {"severity": "moderate", "description": "curated note", "source": "curated"}}}
    ddinter = {"Warfarin": {"Ibuprofen": {"severity": "moderate", "description": "ddinter", "source": "ddinter"}}}
    merged = grounding.merge_interactions(curated, ddinter, db)
    # DDInter pair survives
    assert merged["Warfarin"]["Ibuprofen"]["severity"] == "moderate"
    # ONC forces the nitrate+PDE5 pair to major (overriding the curated 'moderate')
    assert merged["Nitroglycerin"]["Sildenafil"]["severity"] == "major"


# --- citation resolution semantics ---------------------------------------- #
def test_citation_resolves_ignores_nonurl():
    assert grounding.citation_resolves("") is None
    assert grounding.citation_resolves("not-a-url") is None

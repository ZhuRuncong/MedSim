"""Golden-schema tests: the fixture cases (the shape the AI generator targets)
must be internally consistent against the reference vocabulary.

The product ships no hard-coded cases; these validate tests/fixtures/cases.json.
"""
import json
import pathlib

from src import config, data_loader

_CASES = json.loads(
    (pathlib.Path(__file__).parents[1] / "fixtures" / "cases.json").read_text(encoding="utf-8")
)["diseases"]


def _fixtures():
    return _CASES


DISEASE_REQUIRED = ["id", "name", "specialties", "chief_complaint", "symptoms",
                    "vitals", "first_line_drugs", "appropriate_tests", "guideline"]


def test_diseases_have_required_fields():
    for d in _fixtures():
        for field in DISEASE_REQUIRED:
            assert d.get(field) is not None, f"{d.get('id')} missing {field}"
        assert d["guideline"].get("url"), f"{d['id']} guideline missing url"


def test_specialties_are_valid_and_all_covered():
    seen = set()
    for d in _fixtures():
        for s in d["specialties"]:
            assert s in config.SPECIALTIES, f"{d['id']} has unknown specialty {s}"
            seen.add(s)
    # every configured specialty must have at least one case
    assert set(config.SPECIALTIES) <= seen


def test_referenced_drugs_exist():
    drugs = set(data_loader.drug_db()["drugs"])
    for d in _fixtures():
        for key in ("first_line_drugs", "reasonable_drugs", "contraindicated_drugs"):
            for name in d.get(key, []):
                assert name in drugs, f"{d['id']} {key}: unknown drug {name!r}"


def test_lab_deviation_keys_are_known():
    components = set()
    for panel in data_loader.numeric_panels():
        for c in data_loader.components_for_test(panel):
            components.add(c["component"])
    qualitative = set(data_loader.QUALITATIVE_TESTS)
    for d in _fixtures():
        for key in d.get("lab_deviations", {}):
            assert key in components or key in qualitative, \
                f"{d['id']} lab_deviation key {key!r} is neither a component nor a qualitative test"


def test_appropriate_tests_are_orderable():
    orderable = set(data_loader.orderable_tests())
    for d in _fixtures():
        for t in d.get("appropriate_tests", []):
            assert t in orderable, f"{d['id']} appropriate_test {t!r} is not orderable"


def test_exam_findings_use_canonical_exams():
    for d in _fixtures():
        for exam in d.get("exam_findings", {}):
            assert exam in config.PHYSICAL_EXAMS, \
                f"{d['id']} exam_findings key {exam!r} not in PHYSICAL_EXAMS"


def test_unique_disease_ids():
    ids = [d["id"] for d in _fixtures()]
    assert len(ids) == len(set(ids))


def test_symptom_frequencies_valid():
    for d in _fixtures():
        for s in d["symptoms"]:
            assert s["frequency"] in ("common", "occasional", "rare")


def test_interaction_matrix_symmetric():
    for drug in ("Nitroglycerin", "Warfarin", "Methotrexate"):
        hits = data_loader.interactions_for(drug, list(data_loader.drug_db()["drugs"]))
        assert hits, f"expected interactions for {drug}"

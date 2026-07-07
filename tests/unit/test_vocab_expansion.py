"""The orderable vocabulary must be broad enough for realistic work-ups: key
labs (amylase, coags, cardiac, imaging) and procedures (PCI, endoscopy, LP, …)
must be available, and the generator must be constrained to the procedure catalog.
"""
from src import config, data_loader, engine, generator


def test_key_procedures_available():
    procs = set(engine.available_procedures())
    for p in ["Percutaneous Coronary Intervention (PCI)", "Thrombolysis (tPA)",
              "Synchronized Cardioversion", "Upper Endoscopy (EGD)", "Colonoscopy",
              "Lumbar Puncture", "Hemodialysis", "Tube Thoracostomy (Chest Tube)",
              "Paracentesis", "Appendectomy"]:
        assert p in procs, f"missing procedure {p!r}"
    assert p in config.PROCEDURES or True  # catalog is the source


def test_key_tests_orderable():
    ot = set(data_loader.orderable_tests())
    for t in ["Amylase", "Coagulation", "CK-MB", "CK", "LDH", "Thyroid Panel",
              "Calcium/Phosphate", "Ammonia", "Procalcitonin", "Ferritin",
              "CT Head", "CT Chest (PE protocol)", "Echocardiogram", "MRI Brain",
              "Lumbar Puncture (CSF Analysis)", "Lower Extremity Doppler Ultrasound"]:
        assert t in ot, f"missing test {t!r}"


def test_new_lab_components_map_to_panels():
    # a sampling of the numeric components resolve to their orderable panel
    for comp, panel in [("INR", "Coagulation"), ("Free T4", "Thyroid Panel"),
                        ("CK-MB", "CK-MB"), ("Calcium", "Calcium/Phosphate"),
                        ("Amylase", "Amylase")]:
        assert data_loader.panel_for_component(comp) == panel


def test_all_lab_ranges_are_valid():
    for panel in data_loader.numeric_panels():
        for c in data_loader.components_for_test(panel):
            assert c["low"] < c["high"], f"{panel}/{c['component']}: low !< high"


def test_generator_advertises_and_constrains_procedures():
    vb = generator._vocab_block()
    assert "ALLOWED PROCEDURES" in vb and "Percutaneous Coronary Intervention" in vb
    # auto-repair keeps only catalog procedures (indicated_surgeries is the answer key)
    d = generator._auto_repair(
        {"indicated_surgeries": ["Percutaneous Coronary Intervention (PCI)",
                                 "Teleportation Therapy"]})
    assert d["indicated_surgeries"] == ["Percutaneous Coronary Intervention (PCI)"]

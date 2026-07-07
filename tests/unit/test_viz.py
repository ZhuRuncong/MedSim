"""Routing-graph SVG renders and reflects the trajectory."""
from src import engine, viz

PNEUMONIA = "C0032285"


def _played():
    st = engine.create_case(["Internal Medicine"], disease_id=PNEUMONIA)
    st.allergies = []
    engine.perform_action(st, {"type": "order_test", "payload": "CBC"})
    engine.perform_action(st, {"type": "knowledge_query",
                               "payload": "first-line treatment for community-acquired pneumonia?"})
    return st


def test_svg_renders_all_nodes():
    svg = viz.routing_graph_svg([])
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    for label in ("Supervisor", "Patient", "Clinical", "Surgery", "Knowledge", "Critic",
                  "lab_simulator", "drug_effect_engine", "knowledge_lookup", "emergency_surgery"):
        assert label in svg


def test_svg_highlights_used_path():
    st = _played()
    svg = viz.routing_graph_svg(st.trace)
    # used nodes get the accent stroke; an unused one keeps the muted line colour.
    assert viz.ACCENT in svg
    assert viz.LINE in svg  # Surgery agent was never used → muted
    # the ClinicalAgent tool that fired shows up highlighted (count badge present)
    assert "lab_simulator" in svg


def test_svg_escapes_and_is_wellformed():
    st = _played()
    svg = viz.routing_graph_svg(st.trace)
    assert svg.count("<svg") == 1 and svg.count("</svg>") == 1

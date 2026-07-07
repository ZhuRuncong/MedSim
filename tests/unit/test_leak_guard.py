"""The hidden diagnosis must never leak into student-facing text while a case is
active — not through drug/surgery rationales, lab findings, or anywhere else.

Regression: prescribing a first-line drug used to print "… is a guideline
first-line agent for ST-Elevation Myocardial Infarction", handing the student the
answer. The engine now redacts the diagnosis (name + aliases) from every message
until the case closes, and the tools no longer name it at the source.
"""
import re

from src import engine
from src.util import redact_citation, redact_diagnosis


def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# --- redact_diagnosis unit ------------------------------------------------- #
def test_redact_scrubs_name_and_aliases():
    disease = {"name": "ST-Elevation Myocardial Infarction",
               "aliases": ["STEMI", "heart attack"]}
    out = redact_diagnosis("ECG consistent with an acute STEMI / heart attack.", disease)
    assert "STEMI" not in out.upper()
    assert "heart attack" not in out.lower()
    out2 = redact_diagnosis("Foo is first-line for ST-Elevation Myocardial Infarction.", disease)
    assert "Myocardial Infarction" not in out2


def test_redact_is_whole_word_and_empty_safe():
    assert redact_diagnosis("", {"name": "MI", "aliases": []}) == ""
    # whole-word 'MI' is scrubbed; unrelated substrings are left alone
    out = redact_diagnosis("The MI territory looks fine.", {"name": "MI", "aliases": []})
    assert "MI" not in out
    assert "fine" in out


def test_redact_handles_parenthetical_name_forms():
    # name carries an abbreviation in parens; text uses the plain long form
    disease = {"name": "Acute Myocardial Infarction (STEMI)", "aliases": []}
    out = redact_diagnosis("ECG shows acute myocardial infarction pattern.", disease)
    assert "myocardial infarction" not in out.lower()
    # and the abbreviation alone is scrubbed too
    assert "STEMI" not in redact_diagnosis("Looks like a STEMI.", disease).upper()


def test_redact_citation_handles_percent_encoding():
    disease = {"name": "Myocardial Infarction", "aliases": []}
    leaky = "https://example.com/?term=myocardial%20infarction+guideline"
    assert "myocardialinfarction" not in _alnum(redact_citation(leaky, disease))


# --- end-to-end through the engine ----------------------------------------- #
def _mi_case():
    st = engine.create_case(["Emergency"], disease_id="C0155626")  # Acute MI fixture
    st.allergies = []
    return st


def test_prescribing_first_line_does_not_leak_diagnosis():
    st = _mi_case()
    msgs = engine.perform_action(st, {"type": "prescribe_drug", "payload": "Aspirin"})
    blob = " ".join(m.text for m in msgs)
    assert "Myocardial Infarction" not in blob
    assert "STEMI" not in blob.upper()
    assert any(m.points_delta > 0 for m in msgs)  # still credited as first-line


def test_lab_finding_does_not_leak_diagnosis():
    st = _mi_case()  # the ECG finding literally says "consistent with an inferior STEMI"
    msgs = engine.perform_action(st, {"type": "order_test", "payload": "ECG"})
    blob = " ".join(m.text for m in msgs)
    assert "STEMI" not in blob.upper()
    assert "Myocardial Infarction" not in blob


def test_correct_diagnosis_reveal_still_shows_the_name():
    st = _mi_case()
    msgs = engine.perform_action(st, {"type": "submit_diagnosis", "payload": "myocardial infarction"})
    assert st.status == "complete"
    assert any("Myocardial Infarction" in m.text for m in msgs)  # reveal is allowed once closed


# --- citation-URL leak guard ----------------------------------------------- #
def test_redact_citation_neutralises_diagnosis_in_url():
    disease = {"name": "Acute Myocardial Infarction", "aliases": ["stemi", "mi"]}
    # the sanitized-citation fallback embeds the diagnosis with '+'-joined words
    leaky = "https://pubmed.ncbi.nlm.nih.gov/?term=Acute+Myocardial+Infarction+guideline"
    assert "myocardialinfarction" not in _alnum(redact_citation(leaky, disease))
    # slug form
    assert "stemi" not in _alnum(redact_citation("https://acc.org/stemi-guideline", disease))
    # unrelated URL is left intact
    keep = "https://www.acc.org/guidelines/2021"
    assert redact_citation(keep, disease) == keep


def test_no_message_citation_leaks_diagnosis_while_active():
    st = _mi_case()
    for action in ({"type": "order_test", "payload": "ECG"},
                   {"type": "prescribe_drug", "payload": "Aspirin"}):
        msgs = engine.perform_action(st, action)
        for m in msgs:
            for c in m.citations:
                assert "myocardialinfarction" not in _alnum(c)
                assert "stemi" not in _alnum(c)

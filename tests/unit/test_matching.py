"""Regression tests for the substring→whole-word matching fixes (review findings)."""
from src import data_loader
from src.agents import critic, patient
from src.tools import lookup

MI = "C0155626"          # aliases include the 2-char 'mi'
PNEUMONIA = "C0032285"   # aliases include 'cap'
APPENDICITIS = "C0003615"


def d(i):
    return data_loader.get_disease(i)


# --- CriticAgent diagnosis matching --------------------------------------- #
def test_correct_accepts_full_name_and_abbreviation():
    assert critic._is_correct("myocardial infarction", d(MI))
    assert critic._is_correct("I think it's an MI", d(MI))
    assert critic._is_correct("heart attack", d(MI))


def test_correct_rejects_substring_false_positives():
    # 'mi' must NOT match 'migraine'/'vomiting'; 'cap' must not match 'handicap'.
    assert not critic._is_correct("migraine", d(MI))
    assert not critic._is_correct("vomiting", d(MI))
    assert not critic._is_correct("handicap", d(PNEUMONIA))
    # reverse-direction: a fragment of the name must not match
    assert not critic._is_correct("acute", d(APPENDICITIS))
    assert not critic._is_correct("", d(MI))
    assert not critic._is_correct("   ", d(MI))


def test_partial_rejects_blank_but_accepts_real_differential():
    assert not critic._is_partial("", d(APPENDICITIS))
    assert not critic._is_partial("   ", d(APPENDICITIS))
    # Appendicitis differentials include 'Ectopic Pregnancy'
    assert critic._is_partial("ectopic pregnancy", d(APPENDICITIS))


# --- KnowledgeAgent lookup ------------------------------------------------ #
def test_lookup_ignores_casual_abbreviation_collisions():
    assert lookup("mi casa is nice")["source"] == "none"
    assert lookup("put a cap on spending")["source"] == "none"
    assert lookup("what is the capital of France")["source"] == "none"


def test_lookup_matches_real_condition_names():
    assert lookup("management of community-acquired pneumonia")["source"].startswith("local")
    assert lookup("how do I treat cellulitis")["source"].startswith("local")


# --- PatientAgent history topic matching ---------------------------------- #
def test_match_topic_word_boundary():
    # 'arm' (a keyword for 'radiation') must not match inside 'warm'
    assert patient._match_topic("Do you feel warm?", d(MI)) != "radiation"
    # a genuine question still matches
    assert patient._match_topic("Does the pain radiate to your arm?", d(MI)) == "radiation"

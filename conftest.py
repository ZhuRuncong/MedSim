"""Ensure the project root is importable as `src.*` during tests."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "tests", "fixtures", "cases.json")


@pytest.fixture(scope="session", autouse=True)
def _register_fixture_cases():
    """The product ships no hard-coded cases; the suite injects a fixed set of
    fixture cases into the disease registry so tests have known, stable cases."""
    from src import data_loader

    with open(_FIXTURES, "r", encoding="utf-8") as fh:
        for d in json.load(fh)["diseases"]:
            data_loader.register_disease(d)
    yield


@pytest.fixture(autouse=True)
def _hermetic_llm(monkeypatch):
    """Neutralise any real API key from a developer's .env so the suite never
    makes live LLM calls. Tests exercise the LLM paths with an injected
    MockClient instead."""
    from src import config

    monkeypatch.setattr(config, "GOOGLE_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "", raising=False)
    monkeypatch.setattr(config, "CHECK_CITATIONS", False, raising=False)  # no network in tests

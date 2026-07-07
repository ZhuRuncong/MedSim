"""LLM client abstraction (optional).

A uniform ``complete_json()`` over whichever provider is configured (Gemini or
Anthropic). Import is always safe; ``get_client()`` returns ``None`` when no key
is set so callers can fall back to deterministic behaviour. ``MockClient`` lets
the whole generation harness be tested without any network access.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from . import config

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


class LLMUnavailable(RuntimeError):
    """Raised when an LLM is requested but none is configured/importable."""


def _extract_json(text: str) -> str:
    """Best-effort: pull a JSON object out of a model response."""
    text = text.strip()
    m = _JSON_FENCE.search(text) or _JSON_OBJ.search(text)
    return m.group(1) if (m and m.lastindex) else (m.group(0) if m else text)


class BaseClient:
    name = "base"

    def complete_json(self, system: str, user: str, temperature: float = 0.4) -> dict:
        raise NotImplementedError


class GeminiClient(BaseClient):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai  # google-genai SDK (supersedes google-generativeai)
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model

    def complete_json(self, system: str, user: str, temperature: float = 0.4) -> dict:
        cfg = dict(system_instruction=system, temperature=temperature,
                   response_mime_type="application/json")
        # Gemini 2.5 models "think" by default, which is slow for a turn-based UI.
        # Disable it — structured authoring/verification doesn't need extended
        # reasoning, and validation + multi-reviewer voting are the safety net.
        if "2.5" in self._model:
            try:
                cfg["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        resp = self._client.models.generate_content(
            model=self._model, contents=user,
            config=self._types.GenerateContentConfig(**cfg),
        )
        return json.loads(_extract_json(resp.text))


class AnthropicClient(BaseClient):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        import anthropic  # type: ignore

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete_json(self, system: str, user: str, temperature: float = 0.4) -> dict:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system + "\n\nRespond with ONLY one valid JSON object — no prose, no code fences.",
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return json.loads(_extract_json(text))


class MockClient(BaseClient):
    """Returns canned responses in order — for tests and offline demos."""

    name = "mock"

    def __init__(self, responses: List[dict]):
        self._responses = list(responses)
        self.calls: List[tuple] = []

    def complete_json(self, system: str, user: str, temperature: float = 0.4) -> dict:
        self.calls.append((system, user, temperature))
        if not self._responses:
            raise RuntimeError("MockClient ran out of canned responses")
        return self._responses.pop(0)


def get_client(provider: Optional[str] = None) -> Optional[BaseClient]:
    """Return a client for the requested/auto-selected provider, or None."""
    provider = provider or config.LLM_PROVIDER
    if not provider:
        provider = ("gemini" if config.GOOGLE_API_KEY
                    else "anthropic" if config.ANTHROPIC_API_KEY else "")
    try:
        if provider == "gemini" and config.GOOGLE_API_KEY:
            return GeminiClient(config.GOOGLE_API_KEY, config.GEMINI_MODEL)
        if provider == "anthropic" and config.ANTHROPIC_API_KEY:
            return AnthropicClient(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
    except Exception:
        return None
    return None

"""Session persistence (PLAN §4: "stored in Redis under key session:<id>").

Uses Redis when ``REDIS_URL`` is configured *and* ``redis`` is importable;
otherwise falls back transparently to a process-local in-memory dict so the app
runs with zero infrastructure. The public API is identical either way.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

from .config import REDIS_URL
from .state import GameState


class SessionStore:
    """Key/value store for :class:`GameState`, keyed by ``state.redis_key()``."""

    def __init__(self, redis_url: Optional[str] = None):
        self._mem: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._redis = None
        url = redis_url if redis_url is not None else REDIS_URL
        if url:
            try:
                import redis  # type: ignore

                self._redis = redis.Redis.from_url(url, decode_responses=True)
                self._redis.ping()
            except Exception:
                # Any failure (missing lib, unreachable server) → memory fallback.
                self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def save(self, state: GameState) -> None:
        payload = state.model_dump_json()
        key = state.redis_key()
        if self._redis is not None:
            self._redis.set(key, payload)
        else:
            with self._lock:
                self._mem[key] = payload

    def load(self, key: str) -> Optional[GameState]:
        if self._redis is not None:
            payload = self._redis.get(key)
        else:
            with self._lock:
                payload = self._mem.get(key)
        if not payload:
            return None
        return GameState.model_validate_json(payload)

    def delete(self, key: str) -> None:
        if self._redis is not None:
            self._redis.delete(key)
        else:
            with self._lock:
                self._mem.pop(key, None)


# Convenience singleton for the app / API to share.
_default_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    global _default_store
    if _default_store is None:
        _default_store = SessionStore()
    return _default_store

"""OpenAI-compatible chat completion client for optional LLM synthesis."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

import httpx

# Reuse connections across requests (avoids TCP/TLS setup per chat turn).
_client_lock = threading.Lock()
_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _http_client
    with _client_lock:
        if _http_client is None:
            _http_client = httpx.Client()
        return _http_client


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def llm_configured() -> bool:
    """True when LLM synthesis is enabled and an API key is present."""
    if not _env_bool("LLM_ENABLED", False):
        return False
    return bool(os.environ.get("LLM_API_KEY", "").strip())


def _chat_completions_url() -> str:
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return f"{base}/chat/completions"


def _model_name() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def _max_tokens() -> int:
    raw = os.environ.get("LLM_MAX_TOKENS", "1024").strip()
    try:
        n = int(raw)
        return max(64, min(n, 4096))
    except ValueError:
        return 1024


def complete(system: str, user: str, timeout_s: float = 60.0) -> Optional[str]:
    """
    Call the chat completions API. On any failure or empty reply, return None
    so callers can fall back to template output.
    """
    if not llm_configured():
        return None

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        return None

    url = _chat_completions_url()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": _model_name(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": _max_tokens(),
    }

    client = _get_http_client()
    for attempt in range(2):
        try:
            r = client.post(
                url, headers=headers, json=payload, timeout=timeout_s
            )
            if r.status_code >= 500 and attempt == 0:
                continue
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = (choices[0].get("message") or {}).get("content")
            if not msg or not str(msg).strip():
                return None
            return str(msg).strip()
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
            if attempt == 0:
                continue
            return None
    return None

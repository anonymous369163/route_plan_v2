# -*- coding: utf-8 -*-
"""
Central LLM configuration.

All credentials are read from environment variables (see ``.env.example``):

  LLM_BACKEND       openai (default) | openai_compatible | requests
  OPENAI_API_KEY    API key for the selected endpoint
  OPENAI_BASE_URL   optional endpoint override
  OPENAI_LLM_MODEL  optional default chat model override

``requests`` targets OpenAI-compatible endpoints that do not work with the
openai SDK (httpx); calls then go through :func:`requests_chat_completion`.
"""

import os
import re

_LLM_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "openai_compatible": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "requests": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}

LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai").strip().lower()
if LLM_BACKEND not in _LLM_PRESETS:
    LLM_BACKEND = "openai"

_preset = _LLM_PRESETS[LLM_BACKEND]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = (os.environ.get("OPENAI_BASE_URL") or _preset["base_url"]).strip()
DEFAULT_LLM_MODEL = (os.environ.get("OPENAI_LLM_MODEL") or _preset["model"]).strip()

LLM_USE_REQUESTS = LLM_BACKEND == "requests"

if not os.environ.get("OPENAI_BASE_URL") and OPENAI_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL


# ---------------------------------------------------------------------------
# Utilities for backends that emit <think> reasoning tags
# ---------------------------------------------------------------------------

_RE_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
_RE_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks (and truncated <think>… without closing)."""
    text = _RE_THINK_CLOSED.sub("", text)
    text = _RE_THINK_OPEN.sub("", text)
    return text.strip()


_THINK_TOKEN_OVERHEAD = 3

def requests_chat_completion(
    messages: list,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint via *requests* (urllib3 backend).

    Returns the assistant content with <think> tags already stripped.
    Falls back to the active config values when *base_url* / *api_key* / *model*
    are not provided.

    *max_tokens* is automatically scaled up by ``_THINK_TOKEN_OVERHEAD`` to
    accommodate hidden chain-of-thought tokens that some models emit inside
    ``<think>`` blocks (these tokens count against the completion budget but
    are stripped before returning).
    """
    import requests as _req

    _url = (base_url or OPENAI_BASE_URL).rstrip("/") + "/chat/completions"
    _key = api_key or OPENAI_API_KEY
    _model = model or DEFAULT_LLM_MODEL
    effective_max_tokens = max_tokens * _THINK_TOKEN_OVERHEAD
    payload = {
        "model": _model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
        "stream": False,
    }
    resp = _req.post(
        _url,
        json=payload,
        headers={"Authorization": f"Bearer {_key}"},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = (data["choices"][0]["message"].get("content") or "").strip()
    return strip_think_tags(raw)

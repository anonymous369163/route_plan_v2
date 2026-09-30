"""
LLM integration layer: OpenAI client management and JSON call wrapper.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from .utils import extract_json_from_text

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

def maybe_get_client() -> Optional[Any]:
    from config import (  # type: ignore[import]
        OPENAI_API_KEY as _cfg_key,
        OPENAI_BASE_URL as _cfg_base_url,
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or _cfg_key
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or _cfg_base_url
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    except Exception:
        return None


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    fallback: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Call the LLM and parse the response as JSON; fall back on any error."""
    import sys
    from config import LLM_USE_REQUESTS, strip_think_tags  # type: ignore[import]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        if LLM_USE_REQUESTS:
            from config import requests_chat_completion  # type: ignore[import]
            raw_text = requests_chat_completion(
                messages, model=model, temperature=temperature, max_tokens=8000,
            )
        else:
            client = maybe_get_client()
            if client is None:
                print("[LLM WARN] No client available — using fallback (no LLM).",
                      file=sys.stderr, flush=True)
                return fallback()
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=8000,
            )
            raw_text = strip_think_tags(resp.choices[0].message.content or "")

        parsed = extract_json_from_text(raw_text)
        if not isinstance(parsed, dict):
            print("[LLM WARN] Response not valid JSON — using fallback.",
                  file=sys.stderr, flush=True)
            return fallback()
        return parsed
    except Exception as _llm_exc:
        print(f"[LLM WARN] API call failed ({type(_llm_exc).__name__}: "
              f"{str(_llm_exc)[:120]}) — using fallback.",
              file=sys.stderr, flush=True)
        return fallback()



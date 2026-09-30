"""Optional request/response trace boundary used by resumable front ends."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Protocol


class LLMTrace(Protocol):
    def lookup(self, request: Mapping[str, Any]) -> str | None: ...

    def record(
        self,
        request: Mapping[str, Any],
        raw_response: str,
        parse_status: str,
    ) -> None: ...


_ACTIVE_TRACE: ContextVar[LLMTrace | None] = ContextVar("pilot_llm_trace", default=None)


def current_llm_trace() -> LLMTrace | None:
    return _ACTIVE_TRACE.get()


@contextmanager
def capture_llm_trace(trace: LLMTrace | None) -> Iterator[None]:
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)


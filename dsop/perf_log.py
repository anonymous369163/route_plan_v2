# -*- coding: utf-8 -*-
"""
dsop/perf_log.py
────────────────
Wall-clock performance logging utilities.

Public API
----------
_perf_logging_enabled(explicit) -> bool
_PerfSession                    context-manager, writes one log file per run
_PerfSpan                       context-manager for named steps inside a session
"""

import os
import time
from typing import Optional, Sequence


# LOG_DIR is imported from constants to avoid duplicating the path definition.
def _get_log_dir() -> str:
    from dsop.constants import LOG_DIR
    return LOG_DIR


def _perf_logging_enabled(explicit: Optional[bool] = None) -> bool:
    """Return True if ``explicit`` is True, or if env ``TRAINING_DSOP_PERF_LOG`` is set to a truthy token.

    Default (env unset): off. Enable with ``TRAINING_DSOP_PERF_LOG=1`` (or ``true`` / ``yes`` / ``on``).
    """
    if explicit is not None:
        return bool(explicit)
    v = os.environ.get("TRAINING_DSOP_PERF_LOG")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


class _PerfSession:
    """Wall-clock session log under ``LOG_DIR`` (one file per run)."""

    __slots__ = ("label", "meta_lines", "enabled", "_fp", "_t0", "path")

    def __init__(self, label: str, meta_lines: Optional[Sequence[str]] = None, *, enabled: bool = True):
        self.label = label
        self.meta_lines = list(meta_lines or [])
        self.enabled = bool(enabled)
        self._fp = None
        self._t0 = 0.0
        self.path = ""

    def __enter__(self) -> "_PerfSession":
        self._t0 = time.perf_counter()
        if not self.enabled:
            return self
        log_dir = _get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"trainingDSOP_perf_{ts}.log")
        self._fp = open(self.path, "w", encoding="utf-8")
        self._fp.write(f"session={self.label}\n")
        for line in self.meta_lines:
            self._fp.write(f"{line}\n")
        self._fp.write("---\n")
        self._fp.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fp is not None:
            wall = time.perf_counter() - self._t0
            self._fp.write(f"---\nsession_total_wall_s={wall:.6f}\n")
            if exc_type is not None:
                self._fp.write(f"ended_with_exception={exc_type.__name__}: {exc}\n")
            self._fp.close()
            self._fp = None

    def log(self, step_name: str, step_seconds: float) -> None:
        if self._fp is None:
            return
        wall = time.perf_counter() - self._t0
        self._fp.write(f"wall_s={wall:12.4f} step_s={step_seconds:12.6f} | {step_name}\n")
        self._fp.flush()

    def span(self, step_name: str) -> "_PerfSpan":
        return _PerfSpan(self, step_name)

    def note(self, text: str) -> None:
        if self._fp is None:
            return
        wall = time.perf_counter() - self._t0
        self._fp.write(f"wall_s={wall:12.4f} note | {text}\n")
        self._fp.flush()


class _PerfSpan:
    """Named timing span within a _PerfSession; use as a context manager."""

    __slots__ = ("session", "name", "_t0")

    def __init__(self, session: _PerfSession, name: str):
        self.session = session
        self.name = name
        self._t0 = 0.0

    def __enter__(self) -> "_PerfSpan":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.session._fp is None:
            return
        dt = time.perf_counter() - self._t0
        self.session.log(self.name, dt)

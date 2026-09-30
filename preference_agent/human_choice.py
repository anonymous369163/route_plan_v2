"""Optional human-choice boundary shared by non-CLI front ends."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


_VALID_PHASES = {"online", "terminal", "final_decision"}


def _finite_tuple(values: Sequence[float], *, label: str) -> Tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if len(out) != 4 or not np.isfinite(np.asarray(out, dtype=float)).all():
        raise ValueError(f"{label} must contain four finite values")
    return out


@dataclass(frozen=True)
class ChoiceCandidate:
    candidate_id: str
    path: Tuple[Tuple[float, float], ...]
    objectives_raw: Tuple[float, ...]
    objectives_display: Tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = tuple((float(point[0]), float(point[1])) for point in self.path)
        if len(path) < 2 or not np.isfinite(np.asarray(path, dtype=float)).all():
            raise ValueError("candidate path must contain at least two finite points")
        if not str(self.candidate_id):
            raise ValueError("candidate_id must be non-empty")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "objectives_raw",
            _finite_tuple(self.objectives_raw, label="objectives_raw"),
        )
        object.__setattr__(
            self,
            "objectives_display",
            _finite_tuple(self.objectives_display, label="objectives_display"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "path": self.path,
            "objectives_raw": self.objectives_raw,
            "objectives_display": self.objectives_display,
        }


@dataclass(frozen=True)
class ChoiceRequest:
    request_id: str
    candidate_set_hash: str
    phase: str
    iteration: int
    candidates: Tuple[ChoiceCandidate, ...]
    reference_mins: Tuple[float, ...]
    reference_maxs: Tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        phase: str,
        iteration: int,
        candidates: Sequence[ChoiceCandidate],
        reference_mins: Sequence[float],
        reference_maxs: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> "ChoiceRequest":
        phase = str(phase)
        if phase not in _VALID_PHASES:
            raise ValueError(f"unsupported choice phase: {phase}")
        frozen_candidates = tuple(candidates)
        if not frozen_candidates:
            raise ValueError("choice request requires at least one candidate")
        ref_mins = _finite_tuple(reference_mins, label="reference_mins")
        ref_maxs = _finite_tuple(reference_maxs, label="reference_maxs")
        payload = {
            "phase": phase,
            "iteration": int(iteration),
            "candidates": [candidate.fingerprint_payload() for candidate in frozen_candidates],
            "reference_mins": ref_mins,
            "reference_maxs": ref_maxs,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        candidate_set_hash = hashlib.sha256(encoded).hexdigest()
        request_id = f"{phase}-{int(iteration):02d}-{candidate_set_hash[:16]}"
        return cls(
            request_id=request_id,
            candidate_set_hash=candidate_set_hash,
            phase=phase,
            iteration=int(iteration),
            candidates=frozen_candidates,
            reference_mins=ref_mins,
            reference_maxs=ref_maxs,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class ChoiceResponse:
    candidate_index: int
    confidence: int
    reason: str = ""

    def __post_init__(self) -> None:
        if int(self.candidate_index) < 0:
            raise ValueError("candidate_index must be non-negative")
        if not 1 <= int(self.confidence) <= 5:
            raise ValueError("confidence must be between 1 and 5")
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        object.__setattr__(self, "confidence", int(self.confidence))
        object.__setattr__(self, "reason", str(self.reason or ""))


@runtime_checkable
class ChoiceProvider(Protocol):
    def choose(self, request: ChoiceRequest) -> ChoiceResponse:
        """Return one confirmed response for *request*."""


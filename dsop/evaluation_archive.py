"""Passive recording and finite-run Pareto archiving for route evaluations.

The recorder is deliberately independent of the optimiser's random-number
generators and search decisions.  ``compute_path_metrics`` notifies the active
recorder after an evaluation; outside :func:`capture_evaluations` the hook is a
no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from collections import Counter
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np


class ObjectiveConsistencyError(RuntimeError):
    """Raised when the same decision is observed with inconsistent values."""


@dataclass(frozen=True)
class EvaluationRecord:
    """One objective-evaluation request made during an optimiser run."""

    path: Tuple[Tuple[float, float], ...]
    objectives: Tuple[float, ...]
    constraints: Tuple[float, ...]
    source: str


def _path_array(path: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("path must be a two-dimensional sequence of (x, y) points")
    return np.ascontiguousarray(arr)


def _path_key(path: Sequence[Sequence[float]]) -> Tuple[Tuple[int, ...], bytes]:
    arr = _path_array(path)
    return tuple(arr.shape), arr.tobytes()


def _strictly_dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return strict Pareto dominance in the optimiser's all-minimisation space."""
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    return bool(np.all(lhs <= rhs) and np.any(lhs < rhs))


class EvaluationRecorder:
    """Record every evaluation and maintain a passive feasible Pareto archive."""

    def __init__(
        self,
        *,
        feasibility_tolerance: float = 1e-8,
        consistency_rtol: float = 1e-9,
        consistency_atol: float = 1e-12,
    ) -> None:
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.consistency_rtol = float(consistency_rtol)
        self.consistency_atol = float(consistency_atol)
        self.records: List[EvaluationRecord] = []
        self._archive_records: List[EvaluationRecord] = []
        self._unique_by_decision = {}
        self._n_feasible_decisions = 0

    @property
    def archive_records(self) -> Tuple[EvaluationRecord, ...]:
        return tuple(self._archive_records)

    def _is_feasible(self, record: EvaluationRecord) -> bool:
        objectives = np.asarray(record.objectives, dtype=float)
        constraints = np.asarray(record.constraints, dtype=float)
        return bool(
            np.all(np.isfinite(objectives))
            and np.all(np.isfinite(constraints))
            and np.all(constraints <= self.feasibility_tolerance)
        )

    def _assert_duplicate_consistency(
        self,
        previous: EvaluationRecord,
        current: EvaluationRecord,
    ) -> None:
        same_objectives = np.allclose(
            previous.objectives,
            current.objectives,
            rtol=self.consistency_rtol,
            atol=self.consistency_atol,
        )
        same_constraints = np.allclose(
            previous.constraints,
            current.constraints,
            rtol=self.consistency_rtol,
            atol=self.consistency_atol,
        )
        if not (same_objectives and same_constraints):
            raise ObjectiveConsistencyError(
                "The same decision was evaluated with inconsistent objective or constraint values."
            )

    @staticmethod
    def _build_record(
        *,
        path: Sequence[Sequence[float]],
        objectives: Sequence[float],
        constraints: Optional[Sequence[float]],
        source: Optional[str],
    ) -> EvaluationRecord:
        path_arr = _path_array(path)
        objective_arr = np.asarray(objectives, dtype=float).reshape(-1)
        constraint_arr = (
            np.zeros(0, dtype=float)
            if constraints is None
            else np.asarray(constraints, dtype=float).reshape(-1)
        )
        return EvaluationRecord(
            path=tuple((float(point[0]), float(point[1])) for point in path_arr),
            objectives=tuple(float(value) for value in objective_arr),
            constraints=tuple(float(value) for value in constraint_arr),
            source=str(source or "compute_path_metrics"),
        )

    def record(
        self,
        *,
        path: Sequence[Sequence[float]],
        objectives: Sequence[float],
        constraints: Optional[Sequence[float]],
        source: Optional[str] = None,
    ) -> EvaluationRecord:
        record = self._build_record(
            path=path,
            objectives=objectives,
            constraints=constraints,
            source=source,
        )
        self.records.append(record)

        key = _path_key(path)
        previous = self._unique_by_decision.get(key)
        if previous is not None:
            self._assert_duplicate_consistency(previous, record)
            return record

        self._unique_by_decision[key] = record
        if not self._is_feasible(record):
            return record

        self._n_feasible_decisions += 1
        if any(
            _strictly_dominates(existing.objectives, record.objectives)
            for existing in self._archive_records
        ):
            return record

        self._archive_records = [
            existing
            for existing in self._archive_records
            if not _strictly_dominates(record.objectives, existing.objectives)
        ]
        self._archive_records.append(record)
        return record

    def register_snapshot_candidate(
        self,
        *,
        path: Sequence[Sequence[float]],
        objectives: Sequence[float],
        constraints: Optional[Sequence[float]],
        source: str = "population_snapshot",
    ) -> EvaluationRecord:
        """Ensure a reported population member was previously evaluated.

        Existing decisions are consistency-checked without incrementing the
        evaluation count.  A missing decision is registered as a conservative
        fallback for external population observers and test doubles.
        """
        current = self._build_record(
            path=path,
            objectives=objectives,
            constraints=constraints,
            source=source,
        )
        previous = self._unique_by_decision.get(_path_key(path))
        if previous is not None:
            self._assert_duplicate_consistency(previous, current)
            return previous
        return self.record(
            path=path,
            objectives=objectives,
            constraints=constraints,
            source=source,
        )

    def assert_consistent(
        self,
        *,
        path: Sequence[Sequence[float]],
        objectives: Sequence[float],
        constraints: Optional[Sequence[float]],
    ) -> None:
        """Assert that a recomputation matches the recorded decision values."""
        previous = self._unique_by_decision.get(_path_key(path))
        if previous is None:
            raise ObjectiveConsistencyError("The decision is not present in the evaluation record.")
        current = EvaluationRecord(
            path=tuple((float(x), float(y)) for x, y in _path_array(path)),
            objectives=tuple(float(value) for value in np.asarray(objectives, dtype=float).reshape(-1)),
            constraints=tuple(
                float(value)
                for value in (
                    np.zeros(0, dtype=float)
                    if constraints is None
                    else np.asarray(constraints, dtype=float).reshape(-1)
                )
            ),
            source="consistency_assertion",
        )
        self._assert_duplicate_consistency(previous, current)

    def archive_snapshot(self) -> dict:
        records = self._archive_records
        if records:
            F = np.asarray([record.objectives for record in records], dtype=float)
            G = np.asarray([record.constraints for record in records], dtype=float)
        else:
            F = np.zeros((0, 0), dtype=float)
            G = np.zeros((0, 0), dtype=float)
        return {
            "paths": [[tuple(point) for point in record.path] for record in records],
            "F": F,
            "G": G,
            "sources": [record.source for record in records],
            "n_evaluations": len(self.records),
            "n_unique_decisions": len(self._unique_by_decision),
            "n_feasible_decisions": self._n_feasible_decisions,
            "source_counts": dict(Counter(record.source for record in self.records)),
        }

    def feasible_snapshot(self) -> dict:
        """Return unique feasible evaluations, including dominated decisions."""
        records = [
            record
            for record in self._unique_by_decision.values()
            if self._is_feasible(record)
        ]
        if records:
            F = np.asarray([record.objectives for record in records], dtype=float)
            G = np.asarray([record.constraints for record in records], dtype=float)
        else:
            F = np.zeros((0, 0), dtype=float)
            G = np.zeros((0, 0), dtype=float)
        return {
            "paths": [[tuple(point) for point in record.path] for record in records],
            "F": F,
            "G": G,
            "sources": [record.source for record in records],
        }


_ACTIVE_RECORDER: ContextVar[Optional[EvaluationRecorder]] = ContextVar(
    "dsop_active_evaluation_recorder",
    default=None,
)
_ACTIVE_SOURCE: ContextVar[str] = ContextVar(
    "dsop_active_evaluation_source",
    default="compute_path_metrics",
)


@contextmanager
def capture_evaluations(
    recorder: EvaluationRecorder,
    *,
    source: str = "compute_path_metrics",
) -> Iterator[EvaluationRecorder]:
    """Activate ``recorder`` only for the current execution context."""
    recorder_token = _ACTIVE_RECORDER.set(recorder)
    source_token = _ACTIVE_SOURCE.set(str(source))
    try:
        yield recorder
    finally:
        _ACTIVE_SOURCE.reset(source_token)
        _ACTIVE_RECORDER.reset(recorder_token)


@contextmanager
def evaluation_source(source: str) -> Iterator[None]:
    """Annotate evaluations performed by a nested optimiser phase."""
    token = _ACTIVE_SOURCE.set(str(source))
    try:
        yield
    finally:
        _ACTIVE_SOURCE.reset(token)


def record_active_evaluation(
    *,
    path: Sequence[Sequence[float]],
    objectives: Sequence[float],
    constraints: Optional[Sequence[float]],
) -> None:
    """Notify the active recorder, or do nothing when recording is disabled."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        return
    recorder.record(
        path=path,
        objectives=objectives,
        constraints=constraints,
        source=_ACTIVE_SOURCE.get(),
    )


__all__ = [
    "EvaluationRecord",
    "EvaluationRecorder",
    "ObjectiveConsistencyError",
    "capture_evaluations",
    "evaluation_source",
    "record_active_evaluation",
]

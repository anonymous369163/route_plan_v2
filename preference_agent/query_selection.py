# -*- coding: utf-8 -*-
"""Shared candidate-query selection helpers for interactive preference learning."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

try:
    from pymoo.util.reference_direction import select_points_with_maximum_distance
except Exception:  # pragma: no cover - optional dependency guard
    select_points_with_maximum_distance = None  # type: ignore[assignment]

from .scoring import canonicalize_pareto_metrics_batch


def _as_2d(arr: Sequence[Sequence[float]]) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out[None, :]
    return out


def _empty_indices() -> List[int]:
    return []


def _stable_unique(indices: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for idx in indices:
        idx_i = int(idx)
        if idx_i in seen:
            continue
        seen.add(idx_i)
        out.append(idx_i)
    return out


def _max_distance_indices(
    query_f: Sequence[Sequence[float]],
    n_select: int,
    *,
    distance_metrics: Optional[Sequence[Sequence[float]]] = None,
    random_state: Optional[int] = None,
) -> List[int]:
    arr = _as_2d(query_f)
    if arr.size == 0:
        return _empty_indices()
    dist_arr = arr if distance_metrics is None else _as_2d(distance_metrics)
    if len(dist_arr) != len(arr):
        dist_arr = arr
    target_n = min(int(n_select), len(arr))
    if target_n <= 0:
        return _empty_indices()
    if len(arr) <= target_n or select_points_with_maximum_distance is None:
        chosen = list(range(len(arr)))
    else:
        rng = np.random.default_rng(0 if random_state is None else random_state)
        chosen = _stable_unique(
            np.asarray(
                select_points_with_maximum_distance(dist_arr, target_n, random_state=rng),
                dtype=int,
            ).tolist()
        )
    chosen = chosen[:target_n]
    return sorted(chosen, key=lambda idx: (float(arr[idx, 0]), int(idx)))


def _score_candidates(
    utility_model: Any,
    readable_arr: np.ndarray,
    raw_arr: np.ndarray,
) -> Optional[np.ndarray]:
    if utility_model is None:
        return None
    scorer = getattr(utility_model, "score_canonical_batch", None)
    if callable(scorer):
        scores = scorer(readable_arr)
    else:
        scorer = getattr(utility_model, "score_raw_batch", None)
        if callable(scorer):
            scores = scorer(raw_arr)
        else:
            scorer = getattr(utility_model, "score_batch", None)
            if not callable(scorer):
                return None
            scores = scorer(readable_arr)
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if len(arr) != len(raw_arr):
        return None
    return np.nan_to_num(arr, nan=1e9, posinf=1e9, neginf=-1e9)


def _to_preference_space(utility_model: Any, readable_arr: np.ndarray) -> np.ndarray:
    projector = getattr(utility_model, "to_preference_space", None)
    if callable(projector):
        pref = projector(readable_arr)
        pref_arr = np.asarray(pref, dtype=float)
        if pref_arr.ndim == 1:
            pref_arr = pref_arr[None, :]
        if len(pref_arr) == len(readable_arr):
            return np.nan_to_num(pref_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(readable_arr, dtype=float)


def select_candidate_indices(
    query_f: Sequence[Sequence[float]],
    n_select: int,
    *,
    selector_mode: str = "max_distance",
    distance_metrics: Optional[Sequence[Sequence[float]]] = None,
    utility_model: Any = None,
    history_metrics: Optional[Sequence[Sequence[float]]] = None,
    random_state: Optional[int] = None,
    selection_mode: str = "default",
    diagnostic_round: int = 0,
) -> List[int]:
    arr = _as_2d(query_f)
    if arr.size == 0:
        return _empty_indices()
    target_n = min(int(n_select), len(arr))
    if target_n <= 0:
        return _empty_indices()
    if len(arr) <= target_n:
        return list(range(len(arr)))

    mode = str(selector_mode).lower()
    if mode == "simple":
        readable_arr = np.asarray(canonicalize_pareto_metrics_batch(arr), dtype=float)
        scores = _score_candidates(utility_model, readable_arr, arr)
        informative = (
            scores is not None
            and not np.allclose(scores, scores[0], atol=1e-12, rtol=1e-9)
        )
        anchor_idx = int(np.argmin(scores)) if informative else 0
        selected = [anchor_idx]
        pref_space = _to_preference_space(utility_model, readable_arr)
        if pref_space.ndim == 1:
            pref_space = pref_space[None, :]

        history_space = None
        if history_metrics is not None and len(history_metrics) > 0:
            hist_arr = np.asarray(history_metrics, dtype=float)
            if hist_arr.ndim == 1:
                hist_arr = hist_arr[None, :]
            history_space = _to_preference_space(utility_model, hist_arr)

        diff = pref_space[:, None, :] - pref_space[None, :, :]
        dist_mat = np.sqrt(np.sum(diff * diff, axis=2))

        if target_n >= 2:
            best_idx = None
            best_key = None
            for idx in range(len(arr)):
                if idx == anchor_idx:
                    continue
                contrast = float(np.linalg.norm(pref_space[idx] - pref_space[anchor_idx]))
                score_tie = -float(scores[idx]) if informative else 0.0
                key = (contrast, score_tie, -idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            if best_idx is not None:
                selected.append(int(best_idx))

        while len(selected) < target_n:
            best_idx = None
            best_key = None
            for idx in range(len(arr)):
                if idx in selected:
                    continue
                min_dist = float(np.min(dist_mat[idx, selected]))
                hist_dist = 0.0
                if history_space is not None and history_space.size > 0:
                    deltas = history_space - pref_space[idx]
                    hist_dist = float(np.min(np.sqrt(np.sum(deltas * deltas, axis=1))))
                score_tie = -float(scores[idx]) if informative else 0.0
                key = (min_dist, hist_dist, score_tie, -idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(int(best_idx))
        return selected

    if mode != "fcpe_style":
        resolved_distance_metrics = distance_metrics
        if mode == "max_distance" and resolved_distance_metrics is None and utility_model is not None:
            readable_arr = np.asarray(canonicalize_pareto_metrics_batch(arr), dtype=float)
            resolved_distance_metrics = _to_preference_space(utility_model, readable_arr)
        return _max_distance_indices(
            arr,
            target_n,
            distance_metrics=resolved_distance_metrics,
            random_state=random_state,
        )

    readable_arr = np.asarray(canonicalize_pareto_metrics_batch(arr), dtype=float)
    quality_scores = _score_candidates(utility_model, readable_arr, arr)
    if quality_scores is None or np.allclose(quality_scores, quality_scores[0], atol=1e-12, rtol=1e-9):
        return _max_distance_indices(
            arr,
            target_n,
            distance_metrics=distance_metrics,
            random_state=random_state,
        )

    quality_order = np.argsort(quality_scores, kind="mergesort")
    anchor_idx = int(quality_order[0])
    selected = [anchor_idx]
    pref_space = _to_preference_space(utility_model, readable_arr)
    if pref_space.ndim == 1:
        pref_space = pref_space[None, :]

    history_space = None
    if history_metrics is not None and len(history_metrics) > 0:
        hist_arr = np.asarray(history_metrics, dtype=float)
        if hist_arr.ndim == 1:
            hist_arr = hist_arr[None, :]
        if hist_arr.size > 0:
            history_space = _to_preference_space(utility_model, hist_arr)

    diff = pref_space[:, None, :] - pref_space[None, :, :]
    dist_mat = np.sqrt(np.sum(diff * diff, axis=2))

    def pick_objective_probe_candidate(target_dim: int) -> Optional[int]:
        best_idx = None
        best_key = None
        hold_dims = tuple(idx for idx in range(pref_space.shape[1]) if idx != target_dim)
        for idx in range(len(arr)):
            if idx in selected:
                continue
            vary_score = float(abs(pref_space[idx, target_dim] - pref_space[anchor_idx, target_dim]))
            hold_score = float(
                np.linalg.norm(pref_space[idx, list(hold_dims)] - pref_space[anchor_idx, list(hold_dims)])
            ) if hold_dims else 0.0
            key = (vary_score - 0.75 * hold_score, vary_score, -hold_score, -float(quality_scores[idx]), -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        return best_idx

    if str(selection_mode).lower() == "diagnostic":
        if pref_space.shape[1] > 0:
            start_offset = (2 * int(diagnostic_round)) % pref_space.shape[1]
            objective_order = [
                (start_offset + offset) % pref_space.shape[1]
                for offset in range(pref_space.shape[1])
            ]
            for objective_idx in objective_order:
                if len(selected) >= target_n:
                    break
                diag_idx = pick_objective_probe_candidate(int(objective_idx))
                if diag_idx is not None:
                    selected.append(int(diag_idx))
    elif target_n >= 2:
        best_idx = None
        best_key = None
        for idx in range(len(arr)):
            if idx == anchor_idx:
                continue
            contrast = float(np.linalg.norm(pref_space[idx] - pref_space[anchor_idx]))
            key = (contrast, -float(quality_scores[idx]), -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is not None:
            selected.append(int(best_idx))

    while len(selected) < target_n:
        best_idx = None
        best_key = None
        for idx in range(len(arr)):
            if idx in selected:
                continue
            min_dist = float(np.min(dist_mat[idx, selected])) if selected else float("inf")
            hist_dist = 0.0
            if history_space is not None and history_space.size > 0:
                deltas = history_space - pref_space[idx]
                hist_dist = float(np.min(np.sqrt(np.sum(deltas * deltas, axis=1))))
            key = (min_dist, hist_dist, -float(quality_scores[idx]), -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))
    return selected


def select_candidate_rows(
    query_f: Sequence[Sequence[float]],
    n_select: int,
    *,
    selector_mode: str = "max_distance",
    distance_metrics: Optional[Sequence[Sequence[float]]] = None,
    utility_model: Any = None,
    history_metrics: Optional[Sequence[Sequence[float]]] = None,
    random_state: Optional[int] = None,
    selection_mode: str = "default",
    diagnostic_round: int = 0,
) -> np.ndarray:
    arr = _as_2d(query_f)
    indices = select_candidate_indices(
        arr,
        n_select,
        selector_mode=selector_mode,
        distance_metrics=distance_metrics,
        utility_model=utility_model,
        history_metrics=history_metrics,
        random_state=random_state,
        selection_mode=selection_mode,
        diagnostic_round=diagnostic_round,
    )
    if not indices:
        return np.zeros((0, arr.shape[1]), dtype=float)
    return arr[np.asarray(indices, dtype=int)]


if __name__ == "__main__":
    demo_rows = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [8.0, 8.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    class _DemoUtility:
        def score_canonical_batch(self, readable):
            arr = np.asarray(readable, dtype=float)
            return arr[:, 0]

        def to_preference_space(self, readable):
            return np.asarray(readable, dtype=float)

    print(select_candidate_indices(demo_rows, 2, selector_mode="fcpe_style", utility_model=_DemoUtility()))

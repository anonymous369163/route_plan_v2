# -*- coding: utf-8 -*-
"""
dsop/representative.py
──────────────────────
Select a compact representative query set from the Pareto front.

Public API
----------
select_representative_paths(pareto_F, pareto_X, ...) -> (X_list, F_list)

Internal helpers
----------------
_safe_unit_interval         min-max normalise a 1-D array to [0,1]
_pairwise_l2_distances      (N,N) Euclidean distance matrix
_pick_distinct_candidate    pick the best candidate that is far enough from selected
"""

import numpy as np
from typing import List, Optional, Sequence

from dsop.path_metrics import (
    _canonicalize_objective_batch, _normalize_readable_for_scoring, _normalize_weight_vector,
)
from dsop.preference import _raw_objectives_to_preference_space


def _safe_unit_interval(arr: np.ndarray, invert: bool = False) -> np.ndarray:
    """Min-max normalise arr to [0, 1]; optionally flip so lower becomes 1."""
    vec = np.asarray(arr, dtype=float).reshape(-1)
    if vec.size == 0:
        return vec
    lo = float(np.min(vec))
    hi = float(np.max(vec))
    if hi - lo <= 1e-12:
        out = np.zeros_like(vec)
    else:
        out = (vec - lo) / (hi - lo + 1e-12)
    return 1.0 - out if invert else out

# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def _pairwise_l2_distances(features: np.ndarray) -> np.ndarray:
    arr  = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    diff = arr[:, None, :] - arr[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _pick_distinct_candidate(
    ranked_indices: Sequence[int],
    selected: Sequence[int],
    dist_mat: np.ndarray,
    min_dist: float,
) -> Optional[int]:
    """Return the highest-ranked index that is at least min_dist from all selected."""
    ranked       = [int(idx) for idx in ranked_indices]
    selected_set = set(int(idx) for idx in selected)
    for idx in ranked:
        if idx in selected_set:
            continue
        if not selected:
            return idx
        if float(np.min(dist_mat[idx, list(selected)])) >= float(min_dist):
            return idx
    for idx in ranked:
        if idx not in selected_set:
            return idx
    return None


# ─────────────────────────────────────────────
# Main selection function
# ─────────────────────────────────────────────

def select_representative_paths(
    pareto_F,
    pareto_X,
    weights=None,
    n_select=5,
    focus_names=None,
    utility_model=None,
    hypothesis_weights: Optional[Sequence[Sequence[float]]] = None,
    hypothesis_probs: Optional[Sequence[float]] = None,
    belief_confidence: Optional[float] = None,
):
    """Select representatives via a confidence-adaptive disagreement rule.

    The selector intentionally stays compact and paper-friendly:
    1. Pick one high-quality anchor solution.
    2. Estimate preference uncertainty from the current hypothesis belief.
    3. Fill the remaining slots by adaptively balancing quality and
       disagreement, while enforcing a small distance threshold to avoid
       near-duplicate queries.
    4. If no disagreement signal is available, fall back to diverse quality picks.
    """
    _ = focus_names  # Backward-compatible no-op after removing the old focus-extreme heuristic.

    arr = np.asarray(pareto_F, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if len(arr) == 0:
        return [], []
    if len(arr) <= int(n_select):
        return list(pareto_X), list(pareto_F)

    readable   = _canonicalize_objective_batch(arr)
    pref_space = _normalize_readable_for_scoring(readable)
    dist_mat   = _pairwise_l2_distances(pref_space)
    n          = len(pref_space)
    n_obj      = pref_space.shape[1]
    min_dist   = 0.08 if n <= 20 else 0.10

    utility_scores = None
    if utility_model is not None:
        try:
            if hasattr(utility_model, "score_raw_batch"):
                utility_scores = np.asarray(
                    utility_model.score_raw_batch(arr),
                    dtype=float,
                ).reshape(-1)
            else:
                utility_scores = np.asarray(
                    utility_model.score_batch(_raw_objectives_to_preference_space(arr)),
                    dtype=float,
                ).reshape(-1)
        except Exception:
            utility_scores = None

    if utility_scores is not None:
        quality_scores = np.nan_to_num(utility_scores, nan=1e9, posinf=1e9, neginf=-1e9)
    elif weights is not None:
        quality_scores = pref_space.dot(_normalize_weight_vector(weights, n_obj=n_obj))
    else:
        quality_scores = pref_space.mean(axis=1)
    quality_scores = np.asarray(quality_scores, dtype=float).reshape(-1)

    disagreement = np.zeros(n, dtype=float)
    if hypothesis_weights is not None:
        hyp_cols = [
            pref_space.dot(_normalize_weight_vector(hw, n_obj=n_obj))
            for hw in hypothesis_weights
        ]
        if hyp_cols:
            hyp_mat = np.column_stack(hyp_cols)
            probs = None
            if hypothesis_probs is not None:
                probs = np.asarray(hypothesis_probs, dtype=float).reshape(-1)
                if probs.size == hyp_mat.shape[1]:
                    probs = np.clip(probs, 0.0, None)
                    if float(np.sum(probs)) > 1e-12:
                        probs = probs / float(np.sum(probs))
                    else:
                        probs = None
                else:
                    probs = None

            if probs is None:
                disagreement = np.std(hyp_mat, axis=1)
            else:
                mean_scores = hyp_mat.dot(probs)
                disagreement = np.sqrt(
                    np.sum(probs[None, :] * (hyp_mat - mean_scores[:, None]) ** 2, axis=1)
                )

    quality_order = np.argsort(quality_scores)

    # Always keep one plausible anchor route instead of asking only about
    # highly-disagreeing outliers.
    selected: List[int] = [int(quality_order[0])]

    if np.any(disagreement > 1e-8):
        uncertainty = None
        if belief_confidence is not None:
            uncertainty = 1.0 - float(np.clip(float(belief_confidence), 0.0, 1.0))
        elif hypothesis_probs is not None:
            probs = np.asarray(hypothesis_probs, dtype=float).reshape(-1)
            probs = np.clip(probs, 0.0, None)
            if probs.size > 1 and float(np.sum(probs)) > 1e-12:
                probs = probs / float(np.sum(probs))
                entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
                uncertainty = entropy / float(np.log(len(probs)) + 1e-12)
        if uncertainty is None:
            uncertainty = 0.5

        quality_pref = _safe_unit_interval(quality_scores, invert=True)
        disagreement_pref = _safe_unit_interval(disagreement)
        adaptive_pref = (
            (1.0 - uncertainty) * quality_pref
            + uncertainty * disagreement_pref
        )
        # Higher uncertainty -> more disagreement-seeking. Lower uncertainty ->
        # more exploitation around currently high-quality solutions.
        disagree_order = np.lexsort((quality_scores, -adaptive_pref))
        while len(selected) < int(n_select):
            pick = _pick_distinct_candidate(
                disagree_order,
                selected,
                dist_mat,
                min_dist=min_dist,
            )
            if pick is None or pick in selected:
                break
            selected.append(int(pick))
    else:
        while len(selected) < int(n_select):
            pick = _pick_distinct_candidate(
                quality_order,
                selected,
                dist_mat,
                min_dist=min_dist,
            )
            if pick is None or pick in selected:
                break
            selected.append(int(pick))

    selected = list(dict.fromkeys(selected))[: int(n_select)]
    return [pareto_X[i] for i in selected], [pareto_F[i] for i in selected]

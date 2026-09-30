"""
Metrics canonicalization, normalization, weighted scoring, and pairwise
preference-loss functions.  Also hosts metrics-based prompt formatters
(_format_relative_metrics_list, _format_choice_comparison_prompt) that depend
on normalize_metrics and therefore cannot live in utils.py without circular imports.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .constants import DIRECTIONS, OBJECTIVES
from .dm_noise import argmin_with_bt_noise
from .utils import normalize_weights


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------

def canonicalize_pareto_metrics(f_vec: Sequence[float]) -> List[float]:
    """Convert NSGA-II raw objectives to user space.

    Raw NSGA-II: [cost(min), length(min), -safety(min), -smoothness(min)]
    User space:  [cost(min), length(min), safety(max), smoothness(max)]

    Safety and Smoothness are negated so that *higher = better* in user space,
    matching DIRECTIONS = ("min", "min", "max", "max").
    """
    f = list(np.asarray(f_vec, dtype=float))
    return [float(f[0]), float(f[1]), float(-f[2]), float(-f[3])]


def canonicalize_pareto_metrics_batch(metrics: Sequence[Sequence[float]]) -> np.ndarray:
    """Canonicalize raw objectives to user space: [cost, length, safety, smoothness]."""
    arr = np.asarray(metrics, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.column_stack([arr[:, 0], arr[:, 1], -arr[:, 2], -arr[:, 3]])


def canonicalize_best_metrics(best_metrics: Sequence[float]) -> List[float]:
    return canonicalize_pareto_metrics(best_metrics)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_metrics(metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
    """Min-max normalize; higher-is-better objectives are inverted so 0=best."""
    arr = np.asarray(metrics_list, dtype=np.float64)
    denom = arr.max(axis=0) - arr.min(axis=0) + 1e-8
    normed = (arr - arr.min(axis=0)) / denom
    max_mask = np.array([d == "max" for d in DIRECTIONS])
    if np.any(max_mask):
        normed[:, max_mask] = 1.0 - normed[:, max_mask]
    return normed


# ---------------------------------------------------------------------------
# Weighted scoring and path selection
# ---------------------------------------------------------------------------

def compute_weighted_scores(
    metrics_list: Sequence[Sequence[float]], weights: Sequence[float]
) -> np.ndarray:
    return normalize_metrics(metrics_list).dot(
        np.asarray(normalize_weights(weights), dtype=np.float64)
    )


def select_best_path(
    metrics_list: Sequence[Sequence[float]], weights: Sequence[float]
) -> int:
    return int(np.argmin(compute_weighted_scores(metrics_list, weights)))


# ---------------------------------------------------------------------------
# Tchebycheff scalarization (non-linear golden value function)
# ---------------------------------------------------------------------------

def tchebycheff_score(
    metrics_list: Sequence[Sequence[float]],
    weights: Sequence[float],
    ref_mins: Sequence[float] | None = None,
    ref_maxs: Sequence[float] | None = None,
) -> np.ndarray:
    """Tchebycheff scalarization: psi(x) = max_i (f_i^norm - z*_i) / w_i.

    When *ref_mins*/*ref_maxs* are supplied the normalisation uses those
    fixed bounds (necessary so scores are comparable across different
    candidate sets).  Otherwise falls back to set-internal min-max.
    """
    arr = np.asarray(metrics_list, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]

    if ref_mins is not None and ref_maxs is not None:
        mins = np.asarray(ref_mins, dtype=np.float64)
        maxs = np.asarray(ref_maxs, dtype=np.float64)
    else:
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
    denom = maxs - mins + 1e-8
    normed = (arr - mins) / denom

    max_mask = np.array([d == "max" for d in DIRECTIONS])
    if np.any(max_mask):
        normed[:, max_mask] = 1.0 - normed[:, max_mask]

    w = np.asarray(normalize_weights(weights), dtype=np.float64)
    return np.max(normed / (w + 1e-12), axis=1)


def tchebycheff_score_clipped(
    metrics_list: Sequence[Sequence[float]],
    weights: Sequence[float],
    ref_mins: Sequence[float],
    ref_maxs: Sequence[float],
) -> np.ndarray:
    """Tchebycheff scalarization under fixed bounds with clipped normalization.

    This is intended for benchmark/evaluation scoring only: after applying the
    fixed reference min-max transform and max-objective inversion, each
    normalized coordinate is clipped into [0, 1] before the weighted max is
    taken. This prevents out-of-bound candidates from exploding to arbitrarily
    large psi values when they fall outside the reference envelope.
    """
    arr = np.asarray(metrics_list, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    mins = np.asarray(ref_mins, dtype=np.float64)
    maxs = np.asarray(ref_maxs, dtype=np.float64)
    denom = maxs - mins + 1e-8
    normed = (arr - mins) / denom

    max_mask = np.array([d == "max" for d in DIRECTIONS])
    if np.any(max_mask):
        normed[:, max_mask] = 1.0 - normed[:, max_mask]
    normed = np.clip(normed, 0.0, 1.0)

    w = np.asarray(normalize_weights(weights), dtype=np.float64)
    return np.max(normed / (w + 1e-12), axis=1)


def select_best_path_tchebycheff(
    metrics_list: Sequence[Sequence[float]],
    weights: Sequence[float],
    ref_mins: Sequence[float] | None = None,
    ref_maxs: Sequence[float] | None = None,
) -> int:
    return int(np.argmin(tchebycheff_score(metrics_list, weights, ref_mins, ref_maxs)))


def select_best_path_tchebycheff_bt(
    metrics_list: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    noise_kappa: float,
    ref_mins: Sequence[float] | None = None,
    ref_maxs: Sequence[float] | None = None,
) -> int:
    scores = tchebycheff_score(metrics_list, weights, ref_mins, ref_maxs)
    if float(noise_kappa) < 0.0:
        return int(np.argmin(scores))
    return argmin_with_bt_noise(scores, noise_kappa)


def compute_golden_point(
    reference_pool: Sequence[Sequence[float]],
    true_weights: Sequence[float],
) -> tuple:
    """Locate the golden point and normalisation bounds from a reference pool.

    Returns (golden_metrics, ref_mins, ref_maxs) where *golden_metrics* is
    the raw-objective vector of the Pareto-optimal solution that minimises
    the Tchebycheff value under *true_weights*.
    """
    arr = np.asarray(reference_pool, dtype=np.float64)
    ref_mins = arr.min(axis=0)
    ref_maxs = arr.max(axis=0)
    scores = tchebycheff_score(arr, true_weights, ref_mins, ref_maxs)
    best_idx = int(np.argmin(scores))
    return list(arr[best_idx]), ref_mins.tolist(), ref_maxs.tolist()


def approximation_error(
    pareto_front: Sequence[Sequence[float]],
    golden_metrics: Sequence[float],
    ref_mins: Sequence[float],
    ref_maxs: Sequence[float],
) -> float:
    """E(P) = min_{x in P} ||norm(f(x)) - norm(f(x*))||_2."""
    arr = np.asarray(pareto_front, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    mins = np.asarray(ref_mins, dtype=np.float64)
    maxs = np.asarray(ref_maxs, dtype=np.float64)
    denom = maxs - mins + 1e-8

    normed = (arr - mins) / denom
    max_mask = np.array([d == "max" for d in DIRECTIONS])
    if np.any(max_mask):
        normed[:, max_mask] = 1.0 - normed[:, max_mask]

    gold = (np.asarray(golden_metrics, dtype=np.float64) - mins) / denom
    if np.any(max_mask):
        gold[max_mask] = 1.0 - gold[max_mask]

    dists = np.linalg.norm(normed - gold[None, :], axis=1)
    return float(np.min(dists))


def _find_choice_idx(
    all_metrics: Sequence[Sequence[float]], selected_metrics: Sequence[float]
) -> int:
    """Find index of selected_metrics in all_metrics (nearest-neighbour)."""
    arr = np.asarray(all_metrics, dtype=np.float64)
    sel = np.asarray(selected_metrics, dtype=np.float64)
    return int(np.argmin(np.sum((arr - sel) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Pairwise preference loss
# ---------------------------------------------------------------------------

def pairwise_preference_loss(
    X_list: Sequence[Sequence[Sequence[float]]],
    y_list: Sequence[int],
    weights: Sequence[float],
    margin: float = 0.01,
) -> float:
    if not X_list:
        return 0.0
    w = np.asarray(normalize_weights(weights), dtype=float)
    total_loss, total_pairs = 0.0, 0
    for X, y in zip(X_list, y_list):
        normed = normalize_metrics(X)
        chosen = normed[y]
        diff = chosen - normed  # (n, 4)
        dots = np.dot(diff, w)  # (n,)
        mask = np.arange(len(X)) != y
        total_loss += float(np.sum(np.maximum(0.0, dots[mask] + margin)))
        total_pairs += int(np.sum(mask))
    return float(total_loss / total_pairs) if total_pairs else 0.0


# ---------------------------------------------------------------------------
# Round-level and global fit metrics
# ---------------------------------------------------------------------------

def _round_choice_rank_and_gap_fit(
    metrics_list: Sequence[Sequence[float]],
    choice_idx: int,
    weights: Sequence[float],
    gap_scale: float = 0.06,
) -> Tuple[float, float]:
    scores = np.asarray(compute_weighted_scores(metrics_list, weights), dtype=float)
    n = int(scores.size)
    if n <= 1:
        return 1.0, 1.0

    order = np.argsort(scores, kind="mergesort")
    rank = int(np.where(order == choice_idx)[0][0])
    rank_fit = 1.0 - rank / max(n - 1, 1)

    chosen_score = float(scores[choice_idx])
    other_scores = np.delete(scores, choice_idx)
    if other_scores.size == 0:
        return rank_fit, 1.0
    best_alt = float(np.min(other_scores))
    gap = best_alt - chosen_score
    gap_fit = float(1.0 / (1.0 + np.exp(-gap / max(gap_scale, 1e-6))))
    return rank_fit, gap_fit


def _global_gap_fit(
    X_list: Sequence[Sequence[Sequence[float]]],
    y_list: Sequence[int],
    weights: Sequence[float],
) -> float:
    if not X_list:
        return 0.5
    gap_fits = [
        _round_choice_rank_and_gap_fit(X_round, y_round, weights)[1]
        for X_round, y_round in zip(X_list, y_list)
    ]
    return float(np.mean(gap_fits)) if gap_fits else 0.5


def boltzmann_log_likelihood(
    X_list: Sequence[Sequence[Sequence[float]]],
    y_list: Sequence[int],
    weights: Sequence[float],
    tau: float = 0.15,
) -> float:
    """Log-likelihood of observed choices under a Boltzmann (softmax) choice model.

    Model: P(choose i | w, X) = softmax(-w·norm(X) / tau)[i]
    => log P = -w·norm(X)[chosen]/tau - log Σ_j exp(-w·norm(X)[j]/tau)

    Lower tau = more peaked (confident user); higher tau = more random.
    Returns 0.0 when no history.
    """
    if not X_list:
        return 0.0
    w = np.asarray(normalize_weights(weights), dtype=float)
    total_ll = 0.0
    for X_round, y in zip(X_list, y_list):
        normed = normalize_metrics(X_round)           # (n, 4), lower-is-better
        scores = normed.dot(w)                        # (n,), lower = preferred
        logits = -scores / max(float(tau), 1e-6)      # negate: higher logit = more likely
        logits -= logits.max()                        # numerical stability
        log_probs = logits - np.log(np.sum(np.exp(logits)))
        total_ll += float(log_probs[y])
    return total_ll


# ---------------------------------------------------------------------------
# Metrics-based prompt formatters (require normalize_metrics)
# ---------------------------------------------------------------------------

def _format_relative_metrics_list(
    metrics_list: Sequence[Sequence[float]], precision: int = 2
) -> str:
    arr = np.asarray(metrics_list, dtype=np.float64)
    normed = normalize_metrics(arr)
    labels = ("Cost*", "Length*", "SafetyRisk*", "SmoothnessPenalty*")
    lines = []
    for i, row in enumerate(normed, 1):
        lines.append(
            "  "
            + str(i)
            + ". "
            + ", ".join(f"{labels[j]}={row[j]:.{precision}f}" for j in range(len(labels)))
            + "\n"
        )
    return "".join(lines)


def _format_choice_comparison_prompt(
    all_metrics: Sequence[Sequence[float]],
    choice_idx: int,
    precision: int = 2,
    similar_eps: float = 0.05,
) -> str:
    arr = np.asarray(all_metrics, dtype=np.float64)
    normed = normalize_metrics(arr)
    chosen_raw = arr[choice_idx]
    chosen_norm = normed[choice_idx]
    lines = []
    for idx in range(len(arr)):
        if idx == choice_idx:
            continue
        other_raw = arr[idx]
        other_norm = normed[idx]
        parts = []
        for j, name in enumerate(OBJECTIVES):
            if abs(chosen_norm[j] - other_norm[j]) <= similar_eps:
                parts.append(f"{name}=similar")
                continue
            if DIRECTIONS[j] == "min":
                delta = float(chosen_raw[j] - other_raw[j])
                better = delta < 0.0
            else:
                delta = float(chosen_raw[j] - other_raw[j])
                better = delta > 0.0
            delta_text = f"{delta:+.{precision}f}"
            relation = "better" if better else "worse"
            parts.append(f"{name}={relation} ({delta_text})")
        lines.append(f"  Chosen vs Path {idx + 1}: " + ", ".join(parts))
    return "\n".join(lines)

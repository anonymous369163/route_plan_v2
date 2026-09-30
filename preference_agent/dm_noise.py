"""Shared BT-flip decision-noise helpers aligned with benchmark semantics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _draw_uniform(rng: Optional[Any] = None) -> float:
    sampler = np.random if rng is None else rng
    return float(sampler.random())


def bt_flip_should_swap(delta_abs: float, noise_kappa: float, rng: Optional[Any] = None) -> bool:
    """Return whether BT noise flips the current comparison outcome."""
    if float(noise_kappa) < 0.0:
        return False
    delta = abs(float(delta_abs))
    return _draw_uniform(rng) < float(np.exp(-float(noise_kappa) * delta))


def argmin_with_bt_noise(
    scores: np.ndarray | list[float],
    noise_kappa: float,
    rng: Optional[Any] = None,
) -> int:
    """Benchmark-aligned noisy argmin used by FCPE-style best-of-k choice."""
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if len(arr) == 0:
        raise ValueError("scores must be non-empty")
    if float(noise_kappa) < 0.0:
        return int(np.argmin(arr))

    best_idx = int(np.argmin(arr))
    for idx in range(len(arr)):
        if idx == best_idx:
            continue
        delta = abs(float(arr[idx]) - float(arr[best_idx]))
        if bt_flip_should_swap(delta, noise_kappa, rng=rng):
            best_idx = int(idx)
    return best_idx


def pairwise_label_with_bt_noise(
    a_score: float,
    b_score: float,
    noise_kappa: float,
    rng: Optional[Any] = None,
) -> str:
    """Return pairwise preference label compatible with current DM wrappers."""
    a = float(a_score)
    b = float(b_score)
    if float(noise_kappa) < 0.0:
        if np.isclose(a, b, atol=1e-10, rtol=1e-8):
            return "c"
        return "a" if a < b else "b"

    winner = "a" if a <= b else "b"
    loser = "b" if winner == "a" else "a"
    delta = abs(a - b)
    if bt_flip_should_swap(delta, noise_kappa, rng=rng):
        winner, loser = loser, winner
    return winner

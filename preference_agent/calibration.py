"""Behavior-only calibration helpers used by the simplified mainline."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .constants import DIRECTIONS, OBJECTIVES
from .utils import normalize_weights


def _normalize_for_scoring(metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(metrics_list, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    denom = arr.max(axis=0) - arr.min(axis=0) + 1e-8
    normed = (arr - arr.min(axis=0)) / denom
    max_mask = np.array([d == "max" for d in DIRECTIONS])
    if np.any(max_mask):
        normed[:, max_mask] = 1.0 - normed[:, max_mask]
    return normed


def compute_choice_consistency(
    x_history: List[List[List[float]]],
    y_history: List[int],
    weights: Sequence[float],
) -> Tuple[float, List[Dict[str, object]]]:
    w = np.asarray(normalize_weights(weights), dtype=float)
    n_correct = 0
    per_round: List[Dict[str, object]] = []

    for round_idx, (shown, chosen_idx) in enumerate(zip(x_history, y_history), start=1):
        normed = _normalize_for_scoring(shown)
        scores = normed @ w
        predicted_idx = int(np.argmin(scores))
        correct = predicted_idx == chosen_idx
        if correct:
            n_correct += 1

        chosen_metrics = np.asarray(shown[chosen_idx], dtype=float)
        predicted_metrics = np.asarray(shown[predicted_idx], dtype=float)
        detail: List[str] = []
        if not correct:
            for obj_idx, objective in enumerate(OBJECTIVES):
                c_val = float(chosen_metrics[obj_idx])
                p_val = float(predicted_metrics[obj_idx])
                if abs(c_val - p_val) < 1e-6:
                    continue
                direction = DIRECTIONS[obj_idx]
                is_better = (
                    (direction == "max" and c_val > p_val)
                    or (direction == "min" and c_val < p_val)
                )
                detail.append(
                    f"Chosen path was {'better' if is_better else 'worse'} on {objective} "
                    f"({c_val:.3f} vs {p_val:.3f})."
                )

        per_round.append(
            {
                "round": round_idx,
                "predicted": predicted_idx + 1,
                "actual": chosen_idx + 1,
                "correct": correct,
                "detail": detail,
            }
        )

    accuracy = n_correct / max(len(x_history), 1)
    return accuracy, per_round


def build_consistency_feedback(
    x_history: List[List[List[float]]],
    y_history: List[int],
    weights: Sequence[float],
) -> str:
    if not x_history:
        return ""

    accuracy, per_round = compute_choice_consistency(x_history, y_history, weights)
    lines = [
        "Data consistency report for your previous estimate "
        f"{np.round(np.asarray(weights, dtype=float), 3).tolist()}:"
    ]
    for info in per_round:
        if info["correct"]:
            lines.append(
                f"  Round {info['round']}: prediction correct (path {info['actual']})."
            )
            continue
        lines.append(
            f"  Round {info['round']}: predicted path {info['predicted']}, "
            f"but user chose path {info['actual']}."
        )
        for detail in list(info["detail"])[:3]:
            lines.append(f"    - {detail}")
    lines.append(
        f"  Overall consistency: {sum(1 for info in per_round if info['correct'])}/"
        f"{len(per_round)} ({accuracy:.0%})"
    )
    lines.append("")
    return "\n".join(lines) + "\n"

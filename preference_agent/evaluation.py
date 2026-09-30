"""
Evaluation utilities: session-level metric aggregation, optional round image
rendering, and path display helpers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import RoundTrace, SessionConfig
from .scoring import (
    pairwise_preference_loss,
    select_best_path,
    select_best_path_tchebycheff_bt,
    select_best_path_tchebycheff,
    tchebycheff_score,
)
from .utils import (
    cosine_similarity,
    format_metrics,
    l1_error,
    maybe_mkdir,
)


# ---------------------------------------------------------------------------
# Session-level evaluation
# ---------------------------------------------------------------------------

def _eval_session(
    weights: Sequence[float],
    traces: Sequence[RoundTrace],
    X_list: Sequence[Sequence[Sequence[float]]],
    y_list: Sequence[int],
    true_weights: Optional[Sequence[float]],
    dm_noise_tau: float = 0.0,
    noise_kappa: float = -1.0,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    use_deterministic_target = (
        true_weights is not None
        and (dm_noise_tau > 1e-8 or float(noise_kappa) >= 0.0)
    )

    def _target_choice(t: RoundTrace) -> int:
        if use_deterministic_target:
            return select_best_path_tchebycheff(
                t.presented_metrics,
                true_weights,
                ref_mins=t.dm_ref_mins,
                ref_maxs=t.dm_ref_maxs,
            )
        return t.choice_idx

    choice_acc = (
        float(
            np.mean(
                [
                    float(
                        select_best_path_tchebycheff(
                            t.presented_metrics,
                            t.method_weights_after,
                            ref_mins=t.dm_ref_mins,
                            ref_maxs=t.dm_ref_maxs,
                        )
                        == _target_choice(t)
                    )
                    for t in traces
                ]
            )
        )
        if traces
        else 0.0
    )
    pw_loss = pairwise_preference_loss(X_list, y_list, weights)
    w_err = w_cos = None
    if true_weights is not None:
        w_err = l1_error(weights, true_weights)
        w_cos = cosine_similarity(weights, true_weights)
    return choice_acc, pw_loss, w_err, w_cos


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def maybe_render_round_image(
    cfg: SessionConfig,
    iteration: int,
    candidate_paths: Sequence,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Optional[str]:
    if not cfg.save_round_images or not cfg.map_path:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid_map = np.load(cfg.map_path, allow_pickle=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_map, cmap="gray", origin="upper")
        palette = ["red", "blue", "orange", "purple", "cyan"]
        if start:
            plt.scatter(start[0], start[1], s=120, marker="o", c="lime", label="Start")
        if end:
            plt.scatter(end[0], end[1], s=150, marker="*", c="k", label="End")
        for i, path in enumerate(candidate_paths):
            if path:
                xs, ys = zip(*path)
                plt.plot(xs, ys, linewidth=2, color=palette[i % len(palette)], label=f"Path{i+1}")
        plt.legend(loc="upper right")
        plt.axis("off")
        maybe_mkdir(cfg.output_dir)
        sp = os.path.join(cfg.output_dir, f"round_{iteration + 1}_candidates.png")
        plt.savefig(sp, bbox_inches="tight")
        plt.close()
        return sp
    except Exception:
        return None


def choose_path(
    metrics_list: Sequence[Sequence[float]],
    true_weights: Optional[Sequence[float]],
    interactive: bool,
    dm_mode: str = "tchebycheff",
    ref_mins: Optional[Sequence[float]] = None,
    ref_maxs: Optional[Sequence[float]] = None,
    dm_noise_tau: float = 0.0,
    noise_kappa: float = -1.0,
) -> Tuple[int, str]:
    if interactive:
        print("\nPath options:")
        for i, m in enumerate(metrics_list, 1):
            print(f"{i}. {format_metrics(m, precision=2)}")
        choice = (
            max(1, min(len(metrics_list), int(input(f"Choose path (1~{len(metrics_list)}): ").strip())))
            - 1
        )
        return choice, ""
    if true_weights is None:
        raise ValueError("Must provide true_weights in simulation mode")

    use_bt_noise = float(noise_kappa) >= 0.0 and len(metrics_list) > 1
    use_boltzmann = (not use_bt_noise) and dm_noise_tau > 1e-8 and len(metrics_list) > 1

    if dm_mode == "tchebycheff":
        if use_bt_noise:
            best_idx = select_best_path_tchebycheff_bt(
                metrics_list,
                true_weights,
                noise_kappa=noise_kappa,
                ref_mins=ref_mins,
                ref_maxs=ref_maxs,
            )
        elif use_boltzmann:
            scores = tchebycheff_score(
                metrics_list, true_weights, ref_mins=ref_mins, ref_maxs=ref_maxs,
            )
            logits = -scores / dm_noise_tau
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            best_idx = int(np.random.choice(len(scores), p=probs))
        else:
            best_idx = select_best_path_tchebycheff(
                metrics_list, true_weights, ref_mins=ref_mins, ref_maxs=ref_maxs,
            )
    else:
        best_idx = select_best_path(metrics_list, true_weights)
    return best_idx, ""


def print_round_summary(trace: RoundTrace, method: str) -> None:
    print(f"\n[{method}] Round {trace.iteration + 1}")
    print(f"  Choice: {trace.choice_idx + 1}")
    print(
        "  Weights: "
        f"{np.round(trace.method_weights_before, 4).tolist()} -> "
        f"{np.round(trace.method_weights_after, 4).tolist()}"
    )
    print(f"  Focus: {trace.focus}")
    for item in trace.verifier_scores:
        print(f"  {item.get('id', '')}: score={item.get('total_score', 0):.3f}")

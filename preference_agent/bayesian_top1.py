"""Final Top-1 from a non-controlling SLFN posterior ensemble."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from preference_utility import build_pairwise_utility_learner

from .models import SessionResult
from .scoring import canonicalize_pareto_metrics_batch


def select_posterior_rank_top1(member_scores: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Choose the archive item with minimum posterior expected within-member rank."""
    scores = np.asarray(member_scores, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("member_scores must be a non-empty finite member-by-candidate matrix")
    ranks = np.argsort(np.argsort(scores, axis=1, kind="stable"), axis=1, kind="stable")
    normalized_ranks = ranks / float(max(1, scores.shape[1] - 1))
    expected_ranks = normalized_ranks.mean(axis=0)
    selected = int(np.argmin(expected_ranks))
    optimum_indices = np.argmin(scores, axis=1)
    return {
        "selected_index": selected,
        "posterior_expected_ranks": expected_ranks.tolist(),
        "selected_expected_rank": float(expected_ranks[selected]),
        "selected_optimum_probability": float(np.mean(optimum_indices == selected)),
        "member_optimum_indices": optimum_indices.astype(int).tolist(),
        "member_count": int(scores.shape[0]),
    }


def select_archive_rank_top1(
    archive_raw: Sequence[Sequence[float]],
    utility_learner_kwargs: Mapping[str, Any],
    member_params: Sequence[Mapping[str, Any]],
    member_seeds: Sequence[int],
) -> dict[str, Any]:
    """Reconstruct saved SLFN members and select one archive solution."""
    archive = np.asarray(archive_raw, dtype=float)
    if archive.ndim != 2 or not len(archive):
        raise ValueError("posterior Top-1 requires a non-empty final archive")
    if not member_params or len(member_params) != len(member_seeds):
        raise ValueError("SLFN member parameters and seeds must be aligned")

    kwargs = dict(utility_learner_kwargs or {})
    kwargs.pop("rng_seed", None)
    kwargs.pop("member_seeds", None)
    member_scores: list[list[float]] = []
    for seed, saved in zip(member_seeds, member_params):
        learner = build_pairwise_utility_learner(
            variant="benchmark_slfn", rng_seed=int(seed), **kwargs
        ).load_params(dict(saved))
        scores = np.asarray(learner.score_raw_batch(archive), dtype=float).reshape(-1)
        if len(scores) != len(archive) or not np.all(np.isfinite(scores)):
            raise ValueError("SLFN member produced invalid archive scores")
        member_scores.append(scores.tolist())

    posterior = select_posterior_rank_top1(member_scores)
    selected = int(posterior["selected_index"])
    archive_canon = canonicalize_pareto_metrics_batch(archive)
    return {
        **posterior,
        "selected_F_raw_all_min": archive[selected].tolist(),
        "selected_F_canon": archive_canon[selected].tolist(),
    }


def calibrate_session_top1(result: SessionResult) -> dict[str, Any]:
    """Apply the configured single/ensemble SLFN to the final archive."""
    traces = list(result.round_traces or [])
    mode = str(result.utility_ensemble_mode or "off")
    if mode == "shadow_final":
        params = result.shadow_final_utility_params or {}
        member_params = params.get("members")
        member_seeds = params.get("member_seeds")
        prior_member_count = int(params.get("prior_member_count", len(member_seeds or [])))
        model = (
            "mixed_prior_shadow_slfn_ensemble"
            if 0 < prior_member_count < len(member_seeds or [])
            else "behavior_only_shadow_slfn_ensemble"
            if prior_member_count == 0
            else "prior_informed_shadow_slfn_ensemble"
        )
        controls_search = False
    elif mode == "full_pipeline":
        params = result.final_utility_params or {}
        member_params = params.get("members")
        member_seeds = params.get("member_seeds")
        model = "pgcs_coupled_five_member_slfn_ensemble"
        controls_search = True
    elif mode == "off":
        params = result.final_utility_params or {}
        member_params = [params]
        member_seeds = [int(params.get("rng_seed", 42))]
        model = "single_slfn"
        controls_search = True
    else:
        raise ValueError(f"unsupported utility_ensemble_mode for Top-1: {mode}")
    if not isinstance(member_params, list) or not isinstance(member_seeds, list):
        raise ValueError(f"{mode} SLFN member provenance is incomplete")

    archive_raw = (
        result.final_archive_metrics_raw
        if result.final_archive_metrics_raw is not None
        else []
    )
    posterior = select_archive_rank_top1(
        archive_raw,
        result.utility_learner_kwargs or {},
        member_params,
        member_seeds,
    )
    round_weights = list(getattr(result, "learning_query_weights", []) or [])
    if len(round_weights) != len(traces):
        round_weights = [1.0] * len(traces)
    terminal_weight = float(round_weights[-1]) if round_weights else 1.0
    pairwise_constraints = int(
        sum(max(0, len(trace.presented_metrics) - 1) for trace in traces)
    )
    effective_constraints = float(
        sum(
            weight * max(0, len(trace.presented_metrics) - 1)
            for trace, weight in zip(traces, round_weights)
        )
    )
    return {
        **posterior,
        "observed_rounds": int(len(traces)),
        "pairwise_constraints": pairwise_constraints,
        "effective_pairwise_constraints": effective_constraints,
        "terminal_calibration_weight": terminal_weight,
        "terminal_to_first_weight_ratio": float(
            terminal_weight / round_weights[0] if round_weights else 1.0
        ),
        "member_seeds": [int(seed) for seed in member_seeds],
        "prior_member_count": int(
            params.get("prior_member_count", len(member_seeds))
            if mode != "off"
            else 0
        ),
        "posterior_model": model,
        "observation_model": "chosen_vs_each_rejected",
        "decision_rule": "posterior_expected_normalized_rank",
        "controls_search": controls_search,
        "hidden_utility_used": False,
    }

"""Main interactive session loop for the simplified two-mode mainline."""

from __future__ import annotations

import copy
import hashlib
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import build_consistency_feedback
from .constants import DEFAULT_WEIGHTS, OBJECTIVES
from .evaluation import _eval_session, choose_path
from .hypothesis import _generate_structure_only_estimate_llm
from .human_choice import ChoiceCandidate, ChoiceProvider, ChoiceRequest
from .models import RoundTrace, SessionConfig, SessionResult
from .query_selection import select_candidate_indices
from .scoring import canonicalize_best_metrics, canonicalize_pareto_metrics_batch
from .utils import (
    _decode_candidate_path,
    _get_query_candidate_source,
    l1_error,
    normalize_weights,
)

try:
    from dsop import PreferenceGuidance, run_optimizer_continuous
    from dsop.evaluation_archive import EvaluationRecorder
    from dsop.population import _select_population_environment
except Exception:
    PreferenceGuidance = None  # type: ignore[assignment]
    run_optimizer_continuous = None  # type: ignore[assignment]
    EvaluationRecorder = None  # type: ignore[assignment,misc]
    _select_population_environment = None  # type: ignore[assignment]


_LEGACY_PAIR_MIN_CONFIDENCE = 0.50
_LEGACY_PAIR_MIN_SUPPORT = 1
_LEGACY_PAIR_STRENGTH_SHRINK = 0.75

_ORDER_GATE_WARMUP = 2


def _filter_importance_order_by_weights(
    order_pairs: Sequence[Tuple[int, int]],
    learned_weights: Sequence[float],
) -> List[Tuple[int, int]]:
    """Data-consistency filter for LLM importance ordering.

    Retain pair (i, j) — meaning "objective i is more important than j" —
    only when the current learned weight satisfies  w_i >= w_j.
    """
    w = np.asarray(learned_weights, dtype=float)
    n = len(w)
    return [
        (hi, lo) for hi, lo in order_pairs
        if isinstance(hi, int) and isinstance(lo, int)
        and 0 <= hi < n and 0 <= lo < n
        and w[hi] >= w[lo]
    ]


def _cross_pairs() -> List[Tuple[int, int]]:
    return [(i, j) for i in range(len(OBJECTIVES)) for j in range(i + 1, len(OBJECTIVES))]


def _final_pareto_vars_raw_list(last_result: Dict[str, Any]) -> List[List[float]]:
    pareto_x = last_result.get("pareto_X")
    if pareto_x is not None:
        arr = np.asarray(pareto_x, dtype=float)
        if arr.size > 0:
            if arr.ndim == 2:
                return arr.tolist()
            return [arr.reshape(-1).tolist()]
    best_vars = last_result.get("best_vars")
    if best_vars is not None:
        return [np.asarray(best_vars, dtype=float).reshape(-1).tolist()]
    return []


def _build_structure_prior_payload(
    estimate: Optional[Dict[str, Any]],
    confidence: float,
    *,
    use_legacy_mapping: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    pair_priors = np.zeros(len(_cross_pairs()), dtype=float)
    if estimate is None:
        return pair_priors, None

    pair_prior_payload: Dict[str, float] = {}
    interaction_hints = []
    pair_index = {pair: idx for idx, pair in enumerate(_cross_pairs())}
    for item in estimate.get("interaction_hints", []) or []:
        pair = item.get("pair") or []
        if len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        if left not in OBJECTIVES or right not in OBJECTIVES or left == right:
            continue
        idx = pair_index.get(tuple(sorted((OBJECTIVES.index(left), OBJECTIVES.index(right)))))
        if idx is None:
            continue
        strength = float(np.clip(item.get("strength", 0.0), 0.0, 1.0))
        coeff = (
            0.15 * strength * confidence
            if use_legacy_mapping
            else strength
        )
        pair_priors[idx] = coeff
        pair_prior_payload[f"{left}::{right}"] = coeff
        interaction_hints.append(
            {"pair": [left, right], "type": "joint_bottleneck", "strength": strength}
        )

    importance_order_idx: List[Tuple[int, int]] = []
    obj_idx_map = {name: idx for idx, name in enumerate(OBJECTIVES)}
    for item in estimate.get("importance_order", []) or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        higher, lower = str(item[0]), str(item[1])
        if higher in obj_idx_map and lower in obj_idx_map and higher != lower:
            importance_order_idx.append((obj_idx_map[higher], obj_idx_map[lower]))

    payload = {
        "interaction_hints": interaction_hints,
        "importance_order": importance_order_idx,
        "pair_priors": pair_prior_payload,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reasoning": str(estimate.get("reasoning", "")),
        "fallback_used": False,
        "carried_forward": False,
        "source": "llm",
    }
    return pair_priors, payload


def _empty_structure_prior_payload(
    *,
    confidence: float = 0.0,
    reasoning: str = "",
    fallback_used: bool = True,
    carried_forward: bool = False,
    source: str = "fallback",
) -> Dict[str, Any]:
    return {
        "interaction_hints": [],
        "importance_order": [],
        "pair_priors": {},
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reasoning": reasoning,
        "fallback_used": fallback_used,
        "carried_forward": carried_forward,
        "source": source,
    }


def _stabilize_structure_estimate_legacy(
    estimate: Optional[Dict[str, Any]],
    *,
    confidence: float,
    pair_support_counts: Dict[Tuple[str, str], int],
    order_support_counts: Dict[Tuple[str, str], int],
) -> Optional[Dict[str, Any]]:
    if estimate is None:
        return None

    adjusted = copy.deepcopy(estimate)

    pair_high_conf = float(np.clip(confidence, 0.0, 1.0)) >= _LEGACY_PAIR_MIN_CONFIDENCE
    raw_hints = []
    for item in adjusted.get("interaction_hints", []) or []:
        pair = item.get("pair") or []
        if len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        if left not in OBJECTIVES or right not in OBJECTIVES or left == right:
            continue
        key = tuple(sorted((left, right)))
        raw_hints.append(
            {
                "pair": [left, right],
                "type": "joint_bottleneck",
                "strength": float(np.clip(item.get("strength", 0.0), 0.0, 1.0)),
            }
        )
        if pair_high_conf:
            pair_support_counts[key] = pair_support_counts.get(key, 0) + 1

    stable_hints = []
    for item in raw_hints:
        key = tuple(sorted((str(item["pair"][0]), str(item["pair"][1]))))
        if not pair_high_conf or pair_support_counts.get(key, 0) < _LEGACY_PAIR_MIN_SUPPORT:
            continue
        stable_hints.append(
            {
                "pair": list(item["pair"]),
                "type": "joint_bottleneck",
                "strength": float(
                    np.clip(
                        float(item.get("strength", 0.0)) * _LEGACY_PAIR_STRENGTH_SHRINK,
                        0.0,
                        1.0,
                    )
                ),
            }
        )
    adjusted["interaction_hints"] = stable_hints

    raw_order = adjusted.get("importance_order", []) or []
    stable_order = []
    for item in raw_order:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        higher, lower = str(item[0]), str(item[1])
        if higher not in OBJECTIVES or lower not in OBJECTIVES or higher == lower:
            continue
        key = (higher, lower)
        if pair_high_conf:
            order_support_counts[key] = order_support_counts.get(key, 0) + 1
        if pair_high_conf and order_support_counts.get(key, 0) >= _LEGACY_PAIR_MIN_SUPPORT:
            stable_order.append([higher, lower])
    adjusted["importance_order"] = stable_order
    return adjusted


def _make_init_population(all_prev_x: Optional[np.ndarray], target_size: int = 30) -> Optional[np.ndarray]:
    if all_prev_x is None or len(all_prev_x) < 2:
        return None
    n_need = max(0, target_size - len(all_prev_x))
    extra = [
        np.clip(
            all_prev_x[np.random.randint(len(all_prev_x))]
            + np.random.normal(0, 5, size=all_prev_x.shape[1]),
            0,
            None,
        )
        for _ in range(n_need)
    ]
    return np.vstack([all_prev_x] + extra) if extra else all_prev_x


def _select_fixed_round_candidates(
    query_x: Sequence[Any],
    query_f: Sequence[Sequence[float]],
    n_select: int,
    utility_model,
    *,
    selector_mode: str = "fcpe_style",
    history_metrics: Optional[Sequence[Sequence[float]]] = None,
    selection_mode: str = "default",
    diagnostic_round: int = 0,
) -> Tuple[List[Any], List[List[float]]]:
    if n_select <= 0:
        return [], []
    arr = np.asarray(query_f, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.size == 0:
        return [], []

    raw_x = list(query_x)
    raw_f = arr.tolist()
    if len(raw_f) <= n_select:
        return raw_x, raw_f
    selected = select_candidate_indices(
        raw_f,
        n_select=min(int(n_select), len(raw_f)),
        selector_mode=selector_mode,
        utility_model=utility_model,
        history_metrics=history_metrics,
        selection_mode=selection_mode,
        diagnostic_round=diagnostic_round,
    )
    return [raw_x[idx] for idx in selected], [raw_f[idx] for idx in selected]


def _match_selected_indices(
    source_x: Sequence[Any],
    selected_x: Sequence[Any],
    *,
    source_f: Optional[Sequence[Sequence[float]]] = None,
    selected_f: Optional[Sequence[Sequence[float]]] = None,
) -> List[int]:
    """Map selected rows back using objectives to disambiguate lossy X rows."""
    source = [np.asarray(row, dtype=float).reshape(-1) for row in source_x]
    if (source_f is None) != (selected_f is None):
        raise RuntimeError("source and selected objectives must be provided together")
    source_metrics = (
        None
        if source_f is None
        else [np.asarray(row, dtype=float).reshape(-1) for row in source_f]
    )
    selected_metrics = (
        None
        if selected_f is None
        else [np.asarray(row, dtype=float).reshape(-1) for row in selected_f]
    )
    if source_metrics is not None and len(source_metrics) != len(source):
        raise RuntimeError("source objectives are not aligned with source decisions")
    if selected_metrics is not None and len(selected_metrics) != len(selected_x):
        raise RuntimeError("selected objectives are not aligned with selected decisions")
    used: set[int] = set()
    indices: List[int] = []
    for local_index, selected in enumerate(selected_x):
        selected_row = np.asarray(selected, dtype=float).reshape(-1)
        match = next(
            (
                idx
                for idx, source_row in enumerate(source)
                if idx not in used
                and np.array_equal(source_row, selected_row)
                and (
                    source_metrics is None
                    or np.array_equal(
                        source_metrics[idx], selected_metrics[local_index]
                    )
                )
            ),
            None,
        )
        if match is None:
            raise RuntimeError("selected candidate is absent from its query source")
        used.add(match)
        indices.append(match)
    return indices


def _choice_candidate_id(path: Sequence[Sequence[float]], raw_f: Sequence[float]) -> str:
    path_arr = np.asarray(path, dtype=np.float64)
    f_arr = np.asarray(raw_f, dtype=np.float64)
    return hashlib.sha256(path_arr.tobytes() + f_arr.tobytes()).hexdigest()[:24]


def _build_choice_request(
    *,
    phase: str,
    iteration: int,
    selected_x: Sequence[Any],
    selected_f: Sequence[Sequence[float]],
    source_x: Sequence[Any],
    source_f: Optional[Sequence[Sequence[float]]] = None,
    source_paths: Optional[Sequence[Sequence[Sequence[float]]]],
    selected_source_indices: Optional[Sequence[int]] = None,
    start: Tuple[int, int],
    end: Tuple[int, int],
    utility_learner: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> ChoiceRequest:
    raw_f = np.asarray(selected_f, dtype=float)
    if raw_f.ndim == 1:
        raw_f = raw_f[None, :]
    if raw_f.shape[1:] != (4,):
        raise RuntimeError(f"choice objectives must have shape (n, 4), got {raw_f.shape}")
    display_f = canonicalize_pareto_metrics_batch(raw_f)

    if selected_source_indices is None:
        matched_indices = _match_selected_indices(
            source_x,
            selected_x,
            source_f=source_f,
            selected_f=selected_f if source_f is not None else None,
        )
    else:
        matched_indices = [int(index) for index in selected_source_indices]
        if len(matched_indices) != len(selected_x) or any(
            index < 0 or index >= len(source_x) for index in matched_indices
        ):
            raise RuntimeError("selected source indices are not aligned with candidates")
        for selected, source_index in zip(selected_x, matched_indices):
            if not np.array_equal(
                np.asarray(selected, dtype=float).reshape(-1),
                np.asarray(source_x[source_index], dtype=float).reshape(-1),
            ):
                raise RuntimeError("selected source index does not identify its decision row")
    if source_paths is not None:
        if len(source_paths) != len(source_x):
            raise RuntimeError("exact source paths are not aligned with source decisions")
        paths = [
            [tuple(point) for point in source_paths[index]]
            for index in matched_indices
        ]
    else:
        paths = [_decode_candidate_path(row, start, end) for row in selected_x]

    from dsop.map_cache import _get_cached_map
    from dsop.path_metrics import compute_path_metrics

    grid_map = _get_cached_map()
    candidates: List[ChoiceCandidate] = []
    for local_idx, (path, expected_raw, display) in enumerate(
        zip(paths, raw_f, display_f)
    ):
        recomputed_raw = np.asarray(compute_path_metrics(path, grid_map)[0], dtype=float)
        if not np.allclose(
            recomputed_raw,
            expected_raw,
            rtol=1e-9,
            atol=1e-8,
        ):
            raise RuntimeError(
                "candidate path/objective mismatch: "
                f"expected={expected_raw.tolist()}, recomputed={recomputed_raw.tolist()}"
            )
        candidates.append(
            ChoiceCandidate(
                candidate_id=_choice_candidate_id(path, expected_raw),
                path=tuple((float(x), float(y)) for x, y in path),
                objectives_raw=tuple(float(value) for value in expected_raw),
                objectives_display=tuple(float(value) for value in display),
                metadata={"source_index": int(matched_indices[local_idx])},
            )
        )

    ref_min = getattr(utility_learner, "_ref_min", None)
    ref_max = getattr(utility_learner, "_ref_max", None)
    if ref_min is None or ref_max is None:
        raise RuntimeError("human choice requires registered fixed reference bounds")
    return ChoiceRequest.build(
        phase=phase,
        iteration=int(iteration),
        candidates=candidates,
        reference_mins=np.asarray(ref_min, dtype=float).tolist(),
        reference_maxs=np.asarray(ref_max, dtype=float).tolist(),
        metadata=metadata,
    )


def _choose_with_provider(
    provider: ChoiceProvider,
    request: ChoiceRequest,
) -> Tuple[int, str]:
    response = provider.choose(request)
    if response.candidate_index >= len(request.candidates):
        raise ValueError(
            f"choice index {response.candidate_index} is outside "
            f"0..{len(request.candidates) - 1}"
        )
    return int(response.candidate_index), str(response.reason)


def _utility_choice_audit_metadata(
    utility_learner: Any,
    candidate_f: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    scores = np.asarray(
        utility_learner.score_raw_batch(np.asarray(candidate_f, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if len(scores) == 0:
        raise RuntimeError("utility audit requires at least one candidate score")
    clean_scores = np.nan_to_num(scores, nan=1e9, posinf=1e9, neginf=1e9)
    return {
        "utility_params_before_choice": copy.deepcopy(utility_learner.get_params()),
        "model_candidate_scores_before_choice": clean_scores.tolist(),
        "model_predicted_candidate_index": int(np.argmin(clean_scores)),
    }


def _archive_shortlist_snapshot(
    result: Dict[str, Any],
    utility_model: Any,
    *,
    shortlist_size: int,
) -> Dict[str, Any]:
    archive_f = np.asarray(result.get("archive_F"), dtype=float)
    if archive_f.ndim == 1:
        archive_f = archive_f[None, :]
    archive_paths = list(result.get("archive_paths") or [])
    if len(archive_f) == 0 or len(archive_paths) != len(archive_f):
        raise RuntimeError("Final shortlist requires an aligned non-empty archive.")
    scores = np.asarray(utility_model.score_raw_batch(archive_f), dtype=float).reshape(-1)
    if len(scores) != len(archive_f):
        raise RuntimeError("Final utility returned an incompatible archive score vector.")
    scores = np.nan_to_num(scores, nan=1e9, posinf=1e9, neginf=1e9)
    k = min(max(1, int(shortlist_size)), len(archive_f))
    indices = np.argsort(scores, kind="stable")[:k].astype(int).tolist()
    return {
        "archive_indices": indices,
        "metrics_raw": archive_f[np.asarray(indices, dtype=int)].tolist(),
        "paths": [
            [tuple(point) for point in archive_paths[index]] for index in indices
        ],
        "learned_scores": [float(scores[index]) for index in indices],
        "archive_learned_scores": [float(score) for score in scores],
    }


def _build_session_utility_learners(cfg: SessionConfig) -> Tuple[Any, Optional[Any], str, Dict[str, Any]]:
    """Build the active learner and, for shadow mode, its non-controlling ensemble."""
    from preference_utility import build_pairwise_utility_learner  # type: ignore[import]

    mode = str(cfg.utility_ensemble_mode or "off").strip().lower()
    if mode not in {"off", "shadow_final", "full_pipeline"}:
        raise ValueError(
            "utility_ensemble_mode must be one of {'off', 'shadow_final', 'full_pipeline'}"
        )
    if mode != "off" and str(cfg.utility_learner_variant).strip().lower() != "benchmark_slfn":
        raise ValueError(
            "utility_ensemble_mode requires utility_learner_variant='benchmark_slfn'."
        )
    parameter_shadow_configs = dict(cfg.utility_parameter_shadow_configs or {})
    if parameter_shadow_configs and (
        mode != "off"
        or str(cfg.utility_learner_variant).strip().lower() != "benchmark_slfn"
    ):
        raise ValueError(
            "utility_parameter_shadow_configs requires ensemble mode 'off' "
            "and utility_learner_variant='benchmark_slfn'."
        )

    if mode == "off":
        active = build_pairwise_utility_learner(
            variant=cfg.utility_learner_variant,
            n_obj=len(OBJECTIVES),
            **dict(cfg.utility_learner_kwargs or {}),
        )
        shadow = None
        shadow_role = None
        if parameter_shadow_configs:
            from .slfn_shadow_bank import SLFNParameterShadowBank

            shadow = SLFNParameterShadowBank(
                parameter_shadow_configs,
                shared_kwargs=dict(cfg.utility_learner_kwargs or {}),
            )
            shadow_role = "ofat_parameter_bank"
        return active, shadow, mode, {
            "mode": mode,
            "active_utility_role": "configured_single_learner",
            "active_utility_variant": str(cfg.utility_learner_variant),
            "shadow_utility_role": shadow_role,
        }

    ensemble_kwargs = dict(cfg.utility_learner_kwargs or {})
    # The final ensemble is fixed; an explicit single seed may vary only the
    # active controller for the preregistered seed-robustness experiment.
    active_seed = int(ensemble_kwargs.pop("rng_seed", 42))
    ensemble_kwargs.pop("member_seeds", None)
    ensemble_kwargs["n_obj"] = len(OBJECTIVES)
    member_seeds = [42, 43, 44, 45, 46]
    prior_member_count = int(cfg.shadow_prior_member_count)
    if not 0 <= prior_member_count <= len(member_seeds):
        raise ValueError("shadow_prior_member_count must be between 0 and 5")
    if mode != "shadow_final" and prior_member_count != len(member_seeds):
        raise ValueError("shadow_prior_member_count is configurable only in shadow_final mode")

    if mode == "shadow_final":
        active = build_pairwise_utility_learner(
            variant="benchmark_slfn",
            **dict(ensemble_kwargs, rng_seed=active_seed),
        )
        shadow_kwargs = dict(ensemble_kwargs, member_seeds=member_seeds)
        if prior_member_count != len(member_seeds):
            shadow_kwargs["prior_member_count"] = prior_member_count
        shadow = build_pairwise_utility_learner(
            variant="benchmark_slfn_ensemble",
            **shadow_kwargs,
        )
        return active, shadow, mode, {
            "mode": mode,
            "active_utility_role": "configured_seed_single_learner",
            "active_utility_variant": "benchmark_slfn",
            "active_member_seeds": [active_seed],
            "shadow_utility_role": "five_member_ensemble",
            "shadow_utility_variant": "benchmark_slfn_ensemble",
            "shadow_member_seeds": member_seeds,
            "shadow_prior_member_count": prior_member_count,
            "shadow_prior_member_seeds": member_seeds[:prior_member_count],
            "shadow_behavior_only_member_seeds": member_seeds[prior_member_count:],
        }

    active = build_pairwise_utility_learner(
        variant="benchmark_slfn_ensemble",
        **dict(ensemble_kwargs, member_seeds=member_seeds),
    )
    return active, None, mode, {
        "mode": mode,
        "active_utility_role": "five_member_ensemble",
        "active_utility_variant": "benchmark_slfn_ensemble",
        "active_member_seeds": member_seeds,
        "shadow_utility_role": None,
    }


def _apply_utility_event(
    utility_learner: Any,
    shadow_utility_learner: Optional[Any],
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Keep a shadow ensemble trained on the exact active learner event stream."""
    getattr(utility_learner, method_name)(*args, **kwargs)
    if shadow_utility_learner is not None:
        getattr(shadow_utility_learner, method_name)(*args, **kwargs)


def _pad_reference_rows(
    rows: Sequence[Sequence[float]],
    *,
    padding_ratio: float,
) -> np.ndarray:
    """Return reference rows with a fixed symmetric margin around each span."""

    ratio = float(padding_ratio)
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("reference bounds padding ratio must be finite and non-negative")
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("reference rows must be a non-empty finite matrix")
    if ratio == 0.0:
        return arr
    ref_min = arr.min(axis=0)
    ref_max = arr.max(axis=0)
    margin = ratio * (ref_max - ref_min)
    return np.vstack((ref_min - margin, ref_max + margin))


def _apply_archive_shortlist_choice(
    result: Dict[str, Any],
    shortlist: Dict[str, Any],
    *,
    choice_idx: int,
) -> Dict[str, Any]:
    indices = list(shortlist["archive_indices"])
    if not 0 <= int(choice_idx) < len(indices):
        raise RuntimeError("Final DM choice is outside the archive shortlist.")
    archive_idx = int(indices[int(choice_idx)])
    archive_f = np.asarray(result["archive_F"], dtype=float)
    archive_paths = list(result["archive_paths"])
    updated = dict(result)
    updated["best_metrics"] = archive_f[archive_idx].copy()
    archive_g = result.get("archive_G")
    if archive_g is not None and len(np.asarray(archive_g)) == len(archive_f):
        updated["best_constraints"] = np.asarray(archive_g, dtype=float)[archive_idx].copy()
    archive_x = result.get("archive_X")
    updated["best_vars"] = (
        np.asarray(archive_x, dtype=float)[archive_idx].copy()
        if archive_x is not None and len(np.asarray(archive_x)) == len(archive_f)
        else None
    )
    updated["best_path"] = [tuple(point) for point in archive_paths[archive_idx]]
    updated["recommendation_source"] = "evaluation_archive_top5_final_dm_choice"
    updated["recommendation_archive_index"] = archive_idx
    return updated


def _build_biased_warmstart(
    seed_x: np.ndarray,
    seed_f: np.ndarray,
    utility_learner,
    bias_ratio: float = 0.25,
    target_size: int = 100,
) -> np.ndarray:
    x = np.asarray(seed_x, dtype=float)
    f = np.asarray(seed_f, dtype=float)
    if len(x) <= 4:
        return _make_init_population(x, target_size=target_size)
    bias_ratio = float(np.clip(bias_ratio, 0.0, 1.0))
    if bias_ratio <= 1e-12:
        return _make_init_population(x, target_size=target_size)

    pref_scores = np.asarray(utility_learner.score_raw_batch(f), dtype=float).reshape(-1)
    order = np.argsort(pref_scores)
    n_preferred = min(len(x), max(2, int(round(len(x) * bias_ratio))))
    n_seed_bank = min(len(x), max(n_preferred, int(round(0.5 * target_size))))
    preferred = order[:n_preferred].tolist()
    remaining = order[n_preferred:].tolist()
    n_fill = min(len(remaining), max(0, n_seed_bank - n_preferred))
    if n_fill > 0:
        np.random.shuffle(remaining)
        preferred.extend(remaining[:n_fill])
    seed_bank = x[np.array(preferred, dtype=int)]
    return _make_init_population(seed_bank, target_size=target_size)


def _build_verifier_scores_from_dicts(items: Sequence[Dict[str, Any]], feedback_used: bool) -> List[Dict[str, Any]]:
    return [
        {
            "id": item["hypothesis_id"],
            "total_score": item.get("confidence", 0.0),
            "posterior": 1.0,
            "choice_fit": 1.0,
            "history_fit": 1.0,
            "prior_weights": item["prior_weights"],
            "reason": item.get("reason", ""),
            "interaction_hints": item.get("interaction_hints", []),
            "feedback_used": feedback_used,
        }
        for item in items
    ]


def _environment_selection_policy_name(cfg: SessionConfig) -> Optional[str]:
    if str(cfg.preference_injection_mode) == "utility_objective":
        return "single_learned_utility_objective"
    if str(cfg.preference_guidance_policy) == "benchmark_env_consult_only":
        return (
            "consultation_utility_then_standard_crowding"
            if bool(cfg.standard_crowding_between_interactions)
            else "benchmark_ltr_parity"
        )
    return None


def _reselect_archive_with_final_utility(
    last_result: Dict[str, Any],
    utility_model: Any,
    *,
    use_evaluated_feasible: bool = False,
) -> Dict[str, Any]:
    """Select the reported candidate after the final DM update.

    The last interaction occurs at the generation horizon, so its updated
    utility cannot guide another evolutionary generation.  It must still be
    used for the algorithm's actual returned route.  Pareto-based profiles
    read the passive archive; the direct-utility profile must retain all
    unique feasible evaluations, including dominated candidates.  This
    operation is deterministic and consumes no search RNG.
    """
    pool_name = "evaluated_feasible" if use_evaluated_feasible else "archive"
    candidate_f = np.asarray(last_result.get(f"{pool_name}_F"), dtype=float)
    if candidate_f.ndim == 1:
        candidate_f = candidate_f[None, :]
    if len(candidate_f) == 0:
        raise RuntimeError(f"Cannot make the final recommendation from an empty {pool_name} pool.")

    scores = np.asarray(utility_model.score_raw_batch(candidate_f), dtype=float).reshape(-1)
    if len(scores) != len(candidate_f):
        raise RuntimeError(
            f"Final utility returned an incompatible score vector for the {pool_name} pool."
        )
    scores = np.nan_to_num(scores, nan=1e9, posinf=1e9, neginf=1e9)
    best_idx = int(np.argmin(scores))

    candidate_paths = list(last_result.get(f"{pool_name}_paths") or [])
    if len(candidate_paths) != len(candidate_f):
        raise RuntimeError(f"{pool_name} path/objective alignment is incomplete at final selection.")

    updated = dict(last_result)
    updated["best_metrics"] = np.asarray(candidate_f[best_idx], dtype=float).copy()
    candidate_g = last_result.get(f"{pool_name}_G")
    if candidate_g is not None and len(np.asarray(candidate_g)) == len(candidate_f):
        updated["best_constraints"] = np.asarray(candidate_g, dtype=float)[best_idx].copy()
    candidate_x = last_result.get(f"{pool_name}_X")
    updated["best_vars"] = (
        np.asarray(candidate_x, dtype=float)[best_idx].copy()
        if candidate_x is not None and len(np.asarray(candidate_x)) == len(candidate_f)
        else None
    )
    updated["best_path"] = [tuple(point) for point in candidate_paths[best_idx]]
    if use_evaluated_feasible:
        updated["recommendation_source"] = "evaluated_feasible_final_utility_objective"
        updated["recommendation_feasible_index"] = best_idx
        updated.pop("recommendation_archive_index", None)
    else:
        updated["recommendation_source"] = "evaluation_archive_final_utility"
        updated["recommendation_archive_index"] = best_idx
    return updated


def _apply_updated_utility_to_last_selection(
    last_result: Dict[str, Any],
    utility_model: Any,
    *,
    algorithm_profile: str,
) -> Dict[str, Any]:
    """Re-apply the consultation-generation truncation to its saved pool."""
    if _select_population_environment is None or PreferenceGuidance is None:
        raise RuntimeError("DSOP environmental selection is unavailable.")

    context_x = last_result.get("selection_context_X")
    context_f = last_result.get("selection_context_F")
    if context_x is None or context_f is None:
        # Compatibility for mocked/legacy optimizer results.  Production
        # paper runs always expose the saved selection context.
        return dict(last_result)

    x = np.asarray(context_x, dtype=float)
    f = np.asarray(context_f, dtype=float)
    g_raw = last_result.get("selection_context_G")
    g = None if g_raw is None else np.asarray(g_raw, dtype=float)
    pop_size = int(len(np.asarray(last_result["pop_X"], dtype=float)))
    guidance = PreferenceGuidance.from_utility_model(
        utility_model,
        guidance_strength=0.15,
        linearize=False,
        guidance_ratio=1.0,
    )
    selector = _BenchmarkEnvConsultOnlySelector(
        standard_crowding_between_interactions=True
    )
    pop_x, pop_f, pop_g, state = _select_population_environment(
        x,
        f,
        g,
        pop_size=pop_size,
        guidance=guidance,
        algorithm_profile=algorithm_profile,
        environment_selector=selector,
        generation=0,
    )

    updated = dict(last_result)
    updated["pop_X"] = np.asarray(pop_x, dtype=float)
    updated["pop_F"] = np.asarray(pop_f, dtype=float)
    updated["pop_G"] = None if pop_g is None else np.asarray(pop_g, dtype=float)
    feasible_fronts = list(state.get("feasible_fronts") or [])
    if feasible_fronts:
        nd_idx = np.asarray(feasible_fronts[0], dtype=int)
    else:
        nd_idx = np.asarray(state.get("nd_idx", np.arange(len(pop_f))), dtype=int)
    updated["pareto_X"] = np.asarray(pop_x, dtype=float)[nd_idx].copy()
    updated["pareto_F"] = np.asarray(pop_f, dtype=float)[nd_idx].copy()
    updated["post_feedback_environment_selection"] = True
    updated["post_feedback_preference_truncation_applied"] = bool(
        selector.consultation_truncations
    )
    old_rows = {
        tuple(np.round(row, 9).tolist())
        for row in np.asarray(last_result["pop_X"], dtype=float)
    }
    new_rows = {
        tuple(np.round(row, 9).tolist())
        for row in np.asarray(pop_x, dtype=float)
    }
    updated["post_feedback_population_changed"] = old_rows != new_rows
    return updated


class _BenchmarkEnvConsultOnlySelector:
    """Use utility at a consultation boundary and crowding in later generations."""

    def __init__(self, *, standard_crowding_between_interactions: bool = False) -> None:
        self.standard_crowding_between_interactions = bool(
            standard_crowding_between_interactions
        )
        self.consultation_truncations = 0

    def __call__(
        self,
        *,
        generation: Optional[int],
        pop_X,
        pop_F,
        pop_G,
        population_state,
        front_indices,
        slots: int,
        guidance,
        algorithm_profile,
    ) -> Optional[Dict[str, Any]]:
        _ = (pop_X, population_state, algorithm_profile)
        front_idx = np.asarray(front_indices, dtype=int).reshape(-1)
        slots = int(max(0, slots))
        if slots <= 0 or len(front_idx) == 0:
            return {"selected_front_indices": [], "selection_policy": "benchmark_ltr_parity"}

        if int(generation or 0) <= 0:
            utility_model = None if guidance is None else getattr(guidance, "utility_model", None)
            if utility_model is not None:
                front_F = np.asarray(pop_F, dtype=float)[front_idx]
                front_G = None if pop_G is None else np.asarray(pop_G, dtype=float)[front_idx]
                infeasible = None
                if front_G is not None:
                    cv = np.sum(np.maximum(front_G, 0.0), axis=1)
                    infeasible = cv > 1e-8
                scores = np.asarray(
                    utility_model.score_raw_batch(front_F, is_infeasible=infeasible),
                    dtype=float,
                ).reshape(-1)
                if len(scores) == len(front_idx):
                    scores = np.nan_to_num(scores, nan=1e9, posinf=1e9, neginf=1e9)
                    order = np.argsort(scores, kind="stable")[:slots]
                    self.consultation_truncations += 1
                    return {
                        "selected_front_indices": front_idx[order].tolist(),
                        "selection_policy": "benchmark_ltr_parity",
                    }

        if self.standard_crowding_between_interactions:
            # Returning no custom selection delegates to the backbone's
            # standard crowding-distance truncation, as specified by Algorithm 1.
            return None
        return {
            "selected_front_indices": front_idx[:slots].tolist(),
            "selection_policy": "benchmark_ltr_parity",
        }


def _learning_query_plan(
    allocation: str,
    rounds: int,
    consult_on_final_generation: bool,
) -> Tuple[set[int], int]:
    """Return optimizer chunks with learning queries and post-search query count."""
    name = str(allocation).strip().lower()
    if name == "paper_11_plus_1":
        stop = int(rounds) if consult_on_final_generation else max(0, int(rounds) - 1)
        return set(range(stop)), 0
    if name in {
        "terminal_11_plus_1_weighted_top5",
        "terminal_11_plus_1_geometric_top5",
    }:
        if int(rounds) != 12 or not bool(consult_on_final_generation):
            raise ValueError(
                f"{name} requires rounds=12 and "
                "consult_on_final_generation=True"
            )
        return set(range(11)), 1
    if name == "terminal_9_plus_3":
        if int(rounds) != 12 or not bool(consult_on_final_generation):
            raise ValueError(
                "terminal_9_plus_3 requires rounds=12 and "
                "consult_on_final_generation=True"
            )
        return set(range(9)), 3
    if name == "terminal_8_plus_3":
        if int(rounds) != 12 or not bool(consult_on_final_generation):
            raise ValueError(
                "terminal_8_plus_3 requires rounds=12 and "
                "consult_on_final_generation=True"
            )
        return set(range(8)), 3
    raise ValueError(
        f"unsupported learning_query_allocation: {allocation!r}"
    )


def run_proposed_session(
    cfg: SessionConfig,
    start: Tuple[int, int],
    end: Tuple[int, int],
    true_weights: Optional[Sequence[float]],
    interactive: bool,
    choice_provider: Optional[ChoiceProvider] = None,
) -> SessionResult:
    if run_optimizer_continuous is None or PreferenceGuidance is None:
        raise RuntimeError("dsop.run_optimizer_continuous is required for the simplified mainline.")
    if cfg.prior_estimation_mode not in {"utility_only", "structure_only"}:
        raise ValueError("prior_estimation_mode must be one of {'utility_only', 'structure_only'}")
    if cfg.preference_injection_mode not in {"full_tlr", "env_only", "utility_objective"}:
        raise ValueError(
            "preference_injection_mode must be one of "
            "{'full_tlr', 'env_only', 'utility_objective'}"
        )
    if cfg.preference_guidance_policy not in {"current_full_search", "benchmark_env_consult_only"}:
        raise ValueError(
            "preference_guidance_policy must be one of {'current_full_search', 'benchmark_env_consult_only'}"
        )
    if cfg.history_prompt_mode not in {"current", "benchmark_runner_up"}:
        raise ValueError("history_prompt_mode must be one of {'current', 'benchmark_runner_up'}")
    if cfg.objective_context_mode not in {"basic", "formula_card", "formula_card_dmdef"}:
        raise ValueError(
            "objective_context_mode must be one of {'basic', 'formula_card', 'formula_card_dmdef'}"
        )
    if cfg.query_pool_mode not in {"pareto_front", "critical_front"}:
        raise ValueError("query_pool_mode must be one of {'pareto_front', 'critical_front'}")
    if cfg.query_selector_mode not in {"fcpe_style", "simple", "max_distance"}:
        raise ValueError("query_selector_mode must be one of {'fcpe_style', 'simple', 'max_distance'}")
    if float(cfg.dm_noise_tau) < 0.0:
        raise ValueError("dm_noise_tau must be non-negative")
    if int(cfg.final_shortlist_size) <= 0:
        raise ValueError("final_shortlist_size must be positive")
    if not np.isfinite(float(cfg.noise_kappa)):
        raise ValueError("noise_kappa must be finite")
    if float(cfg.noise_kappa) >= 0.0 and float(cfg.dm_noise_tau) > 1e-8:
        warnings.warn(
            "noise_kappa >= 0 takes precedence over dm_noise_tau; dm_noise_tau will be ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    weights = normalize_weights(DEFAULT_WEIGHTS)
    prior_mode = cfg.prior_estimation_mode
    (
        utility_learner,
        shadow_utility_learner,
        utility_ensemble_mode,
        utility_ensemble_provenance,
    ) = _build_session_utility_learners(cfg)
    restart_bias_ratio = float(np.clip(cfg.preference_restart_bias_ratio, 0.0, 1.0))

    x_history: List[List[List[float]]] = []
    y_history: List[int] = []
    traces: List[RoundTrace] = []
    history_metrics_flat: List[List[float]] = []
    feedback_enabled = bool(cfg.use_consistency_feedback) and prior_mode != "utility_only"

    last_result: Dict[str, Any] | None = None
    prev_pop_x = None
    if EvaluationRecorder is None:
        raise RuntimeError("Evaluation recorder is unavailable because the DSOP package failed to import.")
    evaluation_recorder = EvaluationRecorder()
    ref_registered = False

    # If caller supplies pre-computed reference bounds (e.g. from a PSI snapshot
    # or a pooled archive), apply them immediately so the utility learner's
    # normalisation envelope is fixed before the first DM interaction.
    # The in-loop first-round Pareto snapshot is skipped when ref_registered=True.
    _fixed_ref = getattr(cfg, "fixed_reference_bounds", None)
    if _fixed_ref is not None:
        _fmins = _fixed_ref.get("archive_ref_mins")
        _fmaxs = _fixed_ref.get("archive_ref_maxs")
        if _fmins is not None and _fmaxs is not None:
            _apply_utility_event(
                utility_learner,
                shadow_utility_learner,
                "set_reference_from_bounds",
                _fmins,
                _fmaxs,
            )
            ref_registered = True

    final_structure_prior: Optional[Dict[str, Any]] = None
    structure_trace: List[Dict[str, Any]] = []
    structure_valid_rounds: List[int] = []
    active_structure_pairs = np.zeros(len(_cross_pairs()), dtype=float)
    active_structure_payload: Optional[Dict[str, Any]] = None
    use_legacy_structure_mapping = bool(cfg.use_legacy_structure_mapping)
    legacy_pair_support_counts: Dict[Tuple[str, str], int] = {}
    legacy_order_support_counts: Dict[Tuple[str, str], int] = {}
    utility_objective_mode = cfg.preference_injection_mode == "utility_objective"
    benchmark_env_policy = (
        str(cfg.preference_guidance_policy) == "benchmark_env_consult_only"
        and not utility_objective_mode
    )
    if utility_objective_mode:
        algorithm_profile = "utility_scalarized"
    elif benchmark_env_policy:
        algorithm_profile = "preference_benchmark_env_only"
    elif cfg.preference_injection_mode == "env_only":
        algorithm_profile = "preference_env_only"
    else:
        algorithm_profile = None
    last_query_pool_source: Optional[str] = None
    last_query_pool_generation: Optional[int] = None
    post_feedback_selection_applied_rounds: List[int] = []
    post_feedback_selection_changed_rounds: List[int] = []
    environment_selection_policy = _environment_selection_policy_name(cfg)
    learning_query_sources: List[str] = []
    learning_query_phases: List[str] = []
    learning_query_generations: List[int] = []
    online_query_iterations, terminal_query_count = _learning_query_plan(
        cfg.learning_query_allocation,
        cfg.rounds,
        cfg.consult_on_final_generation,
    )
    weighted_terminal_top5 = (
        str(cfg.learning_query_allocation).strip().lower()
        == "terminal_11_plus_1_weighted_top5"
    )
    geometric_terminal_top5 = (
        str(cfg.learning_query_allocation).strip().lower()
        == "terminal_11_plus_1_geometric_top5"
    )
    terminal_archive_top5 = weighted_terminal_top5 or geometric_terminal_top5
    if terminal_archive_top5 and int(cfg.query_n_select) != 5:
        raise ValueError(f"{cfg.learning_query_allocation} requires query_n_select=5")
    if geometric_terminal_top5 and str(cfg.utility_learner_variant) != "benchmark_slfn":
        raise ValueError(
            "terminal_11_plus_1_geometric_top5 requires utility_learner_variant='benchmark_slfn'"
        )
    temporal_weight_ratio = float(cfg.temporal_weight_ratio)
    if not np.isfinite(temporal_weight_ratio) or temporal_weight_ratio < 1.0:
        raise ValueError("temporal_weight_ratio must be finite and at least 1")

    def perform_learning_query(
        query_x: np.ndarray,
        query_f: np.ndarray,
        *,
        query_pool_source: str,
        query_generation: int,
        learning_iteration: int,
        phase: str,
        allow_search_side_effects: bool,
    ) -> None:
        nonlocal active_structure_pairs
        nonlocal active_structure_payload
        nonlocal final_structure_prior
        nonlocal last_query_pool_generation
        nonlocal last_query_pool_source
        nonlocal last_result
        nonlocal prev_pop_x
        nonlocal weights

        if last_result is None:
            raise RuntimeError("a learning query requires an optimizer result")
        learning_query_sources.append(str(query_pool_source))
        learning_query_phases.append(str(phase))
        learning_query_generations.append(int(query_generation))
        last_query_pool_source = str(query_pool_source)
        last_query_pool_generation = int(query_generation)

        is_terminal_archive_calibration = (
            terminal_archive_top5
            and str(phase) == "terminal"
            and str(query_pool_source) == "evaluation_archive"
        )
        if is_terminal_archive_calibration:
            terminal_model = shadow_utility_learner or utility_learner
            terminal_shortlist = _archive_shortlist_snapshot(
                last_result,
                terminal_model,
                shortlist_size=cfg.query_n_select,
            )
            terminal_indices = terminal_shortlist["archive_indices"]
            selected_x = [query_x[idx] for idx in terminal_indices]
            selected_f = terminal_shortlist["metrics_raw"]
        else:
            selection_mode = "diagnostic" if learning_iteration < 2 else "default"
            selected_x, selected_f = _select_fixed_round_candidates(
                query_x,
                query_f,
                n_select=cfg.query_n_select,
                utility_model=utility_learner,
                selector_mode=cfg.query_selector_mode,
                history_metrics=history_metrics_flat,
                selection_mode=selection_mode,
                diagnostic_round=learning_iteration,
            )

        shown = canonicalize_pareto_metrics_batch(np.asarray(selected_f)).tolist()
        if choice_provider is None:
            choice_idx, user_reason = choose_path(
                shown,
                true_weights,
                interactive,
                dm_mode="tchebycheff",
                ref_mins=(
                    None
                    if utility_learner._ref_min is None
                    else np.asarray(utility_learner._ref_min, dtype=float).tolist()
                ),
                ref_maxs=(
                    None
                    if utility_learner._ref_max is None
                    else np.asarray(utility_learner._ref_max, dtype=float).tolist()
                ),
                dm_noise_tau=cfg.dm_noise_tau,
                noise_kappa=cfg.noise_kappa,
            )
        else:
            exact_paths = (
                last_result.get("archive_paths")
                if str(query_pool_source) == "evaluation_archive"
                else None
            )
            request = _build_choice_request(
                phase=str(phase),
                iteration=int(learning_iteration),
                selected_x=selected_x,
                selected_f=selected_f,
                source_x=query_x,
                source_f=query_f,
                source_paths=exact_paths,
                start=start,
                end=end,
                utility_learner=utility_learner,
                metadata={
                    "query_pool_source": str(query_pool_source),
                    "query_generation": int(query_generation),
                    **_utility_choice_audit_metadata(utility_learner, selected_f),
                },
            )
            choice_idx, user_reason = _choose_with_provider(choice_provider, request)
        x_history.append(copy.deepcopy(shown))
        y_history.append(choice_idx)
        fit_x_history = x_history
        fit_y_history = y_history
        fit_round_weights = None
        if is_terminal_archive_calibration:
            if weighted_terminal_top5:
                fit_x_history = x_history[:-1] + [x_history[-1]] * 5
                fit_y_history = y_history[:-1] + [y_history[-1]] * 5
            else:
                fit_round_weights = np.power(
                    temporal_weight_ratio,
                    np.linspace(0.0, 1.0, len(x_history)),
                )
                fit_round_weights *= len(fit_round_weights) / float(
                    np.sum(fit_round_weights)
                )
        update_kwargs = {
            "round_idx": learning_iteration,
            "total_rounds": cfg.rounds,
        }
        if fit_round_weights is not None:
            update_kwargs["round_weights"] = fit_round_weights.tolist()

        use_behav = getattr(cfg, "use_behavior_diagnostics", True)
        feedback_text = ""
        if feedback_enabled and use_behav:
            feedback_text = build_consistency_feedback(x_history, y_history, weights)

        agg_prior = None
        posterior_confidence = None
        verifier_scores: List[Dict[str, Any]]
        structure_prior_round: Optional[Dict[str, Any]] = None
        focus: List[str] = []

        if prior_mode == "utility_only":
            _apply_utility_event(
                utility_learner,
                shadow_utility_learner,
                "update",
                fit_x_history,
                fit_y_history,
                **update_kwargs,
            )
            verifier_scores = []
            new_weights = normalize_weights(utility_learner.get_linear_weights())
        elif prior_mode == "structure_only":
            structured = _generate_structure_only_estimate_llm(
                shown,
                shown[choice_idx],
                cfg,
                traces=traces,
                consistency_feedback=feedback_text,
            )
            if structured is not None:
                posterior_confidence = float(
                    np.clip(structured.get("confidence", 0.5), 0.0, 1.0)
                )
                if use_legacy_structure_mapping:
                    structured = _stabilize_structure_estimate_legacy(
                        structured,
                        confidence=posterior_confidence,
                        pair_support_counts=legacy_pair_support_counts,
                        order_support_counts=legacy_order_support_counts,
                    )
                active_structure_pairs, structure_prior_round = _build_structure_prior_payload(
                    structured,
                    posterior_confidence,
                    use_legacy_mapping=use_legacy_structure_mapping,
                )
                active_structure_payload = copy.deepcopy(structure_prior_round)
                structure_valid_rounds.append(learning_iteration)
            else:
                if active_structure_payload is None:
                    active_structure_pairs = np.zeros(len(_cross_pairs()), dtype=float)
                    structure_prior_round = _empty_structure_prior_payload(
                        confidence=0.0,
                        reasoning="no_valid_structure_prior",
                        fallback_used=True,
                        carried_forward=False,
                        source="none",
                    )
                else:
                    structure_prior_round = copy.deepcopy(active_structure_payload)
                    structure_prior_round["fallback_used"] = True
                    structure_prior_round["carried_forward"] = True
                    structure_prior_round["source"] = "carry_forward"
                posterior_confidence = float(
                    np.clip(structure_prior_round.get("confidence", 0.0), 0.0, 1.0)
                )

            for hint in structure_prior_round.get("interaction_hints", []):
                focus.extend(
                    [
                        str(value)
                        for value in hint.get("pair", [])
                        if str(value) in OBJECTIVES
                    ]
                )
            for order_pair in structure_prior_round.get("importance_order", []):
                if isinstance(order_pair, (list, tuple)) and len(order_pair) == 2:
                    for idx_val in order_pair:
                        name = (
                            OBJECTIVES[idx_val]
                            if isinstance(idx_val, int) and 0 <= idx_val < len(OBJECTIVES)
                            else str(idx_val)
                        )
                        if name in OBJECTIVES and name not in focus:
                            focus.append(name)
            focus = list(dict.fromkeys(focus))
            verifier_scores = _build_verifier_scores_from_dicts(
                [
                    {
                        "hypothesis_id": "structure_only",
                        "prior_weights": list(normalize_weights(weights)),
                        "focus": focus,
                        "interaction_hints": structure_prior_round.get(
                            "interaction_hints", []
                        ),
                        "confidence": posterior_confidence,
                        "reason": structure_prior_round.get("reasoning", ""),
                    }
                ],
                feedback_enabled,
            )
            structure_pair_scale = 1.0 if use_legacy_structure_mapping else None
            _apply_utility_event(
                utility_learner,
                shadow_utility_learner,
                "set_structure_prior",
                pair_priors=active_structure_pairs,
                confidence=posterior_confidence or 0.0,
                pair_prior_scale=structure_pair_scale,
            )
            raw_order = structure_prior_round.get("importance_order", [])
            if learning_iteration >= _ORDER_GATE_WARMUP:
                raw_order = _filter_importance_order_by_weights(raw_order, weights)
            _apply_utility_event(
                utility_learner,
                shadow_utility_learner,
                "set_importance_order",
                order_pairs=raw_order,
                confidence=posterior_confidence or 0.0,
            )
            _apply_utility_event(
                utility_learner,
                shadow_utility_learner,
                "update",
                fit_x_history,
                fit_y_history,
                **update_kwargs,
            )
            new_weights = normalize_weights(utility_learner.get_linear_weights())
            final_structure_prior = copy.deepcopy(
                active_structure_payload or structure_prior_round
            )

        trace = RoundTrace(
            iteration=learning_iteration,
            presented_metrics=copy.deepcopy(shown),
            choice_idx=choice_idx,
            user_reason=user_reason,
            method_weights_before=list(weights),
            method_weights_after=list(new_weights),
            focus=focus,
            verifier_scores=verifier_scores,
            utility_params=utility_learner.get_params(),
            agg_prior_weights=list(agg_prior) if agg_prior is not None else None,
            posterior_confidence=posterior_confidence,
            prior_strategy=prior_mode,
            feedback_used=feedback_enabled,
            lambda_prior_used=utility_learner.current_lambda_prior,
            cross_term_active=utility_learner.has_active_cross_terms(),
            structure_prior=copy.deepcopy(structure_prior_round),
            dm_ref_mins=(
                np.asarray(utility_learner._ref_min).tolist()
                if utility_learner._ref_min is not None
                else None
            ),
            dm_ref_maxs=(
                np.asarray(utility_learner._ref_max).tolist()
                if utility_learner._ref_max is not None
                else None
            ),
        )
        traces.append(trace)

        if structure_prior_round is not None:
            structure_trace.append(
                {
                    "iteration": learning_iteration,
                    "prior_strategy": prior_mode,
                    "llm_structure_valid": bool(
                        prior_mode == "structure_only"
                        and learning_iteration in structure_valid_rounds
                    ),
                    "structure_prior": copy.deepcopy(structure_prior_round),
                }
            )

        history_metrics_flat.extend([list(metrics) for metrics in shown])
        weights = list(new_weights)

        if (
            allow_search_side_effects
            and benchmark_env_policy
            and cfg.inject_preference_guidance
            and cfg.standard_crowding_between_interactions
        ):
            last_result = _apply_updated_utility_to_last_selection(
                last_result,
                utility_learner,
                algorithm_profile=str(algorithm_profile),
            )
            if last_result.get("post_feedback_preference_truncation_applied"):
                post_feedback_selection_applied_rounds.append(learning_iteration)
            if last_result.get("post_feedback_population_changed"):
                post_feedback_selection_changed_rounds.append(learning_iteration)
            prev_pop_x = np.asarray(last_result["pop_X"], dtype=float)

        if (
            allow_search_side_effects
            and cfg.use_preference_restart
            and cfg.preference_injection_mode != "env_only"
            and prev_pop_x is not None
            and utility_learner._ref_min is not None
            and learning_iteration > 0
            and learning_iteration < cfg.rounds - 1
        ):
            seed_x = np.asarray(last_result.get("pareto_X", prev_pop_x), dtype=float)
            seed_f = np.asarray(
                last_result.get("pareto_F", last_result["pop_F"]), dtype=float
            )
            prev_pop_x = _build_biased_warmstart(
                seed_x,
                seed_f,
                utility_learner,
                bias_ratio=restart_bias_ratio,
                target_size=len(np.asarray(prev_pop_x, dtype=float)),
            )

    for iteration in range(cfg.rounds):
        n_gen = cfg.n_gen_warmup if iteration == 0 else cfg.n_gen_per_interaction
        guidance = None
        if cfg.inject_preference_guidance and iteration > 0:
            guidance_ratio = (
                1.0
                if cfg.preference_injection_mode in {"env_only", "utility_objective"}
                else float(cfg.guidance_ratio)
            )
            guidance_strength = 0.15 if cfg.fixed_guidance_strength is None else float(cfg.fixed_guidance_strength)
            guidance = PreferenceGuidance.from_utility_model(
                utility_learner,
                guidance_strength=guidance_strength,
                linearize=False,
                guidance_ratio=guidance_ratio,
                utility_as_objective=(cfg.preference_injection_mode == "utility_objective"),
            )
        environment_selector = None
        if (
            benchmark_env_policy
            and iteration > 0
            and cfg.inject_preference_guidance
            and not cfg.standard_crowding_between_interactions
        ):
            environment_selector = _BenchmarkEnvConsultOnlySelector(
                standard_crowding_between_interactions=bool(
                    cfg.standard_crowding_between_interactions
                )
            )

        last_result = run_optimizer_continuous(
            user_weights=weights,
            start=start,
            end=end,
            n_gen=n_gen,
            guidance=guidance,
            prev_pop_X=prev_pop_x,
            algorithm_profile=algorithm_profile,
            environment_selector=environment_selector,
            evaluation_recorder=evaluation_recorder,
        )
        prev_pop_x = last_result["pop_X"]

        if iteration not in online_query_iterations:
            continue

        query_x, query_f, query_pool_source, query_pool_generation = (
            _get_query_candidate_source(
                last_result,
                query_pool_mode=cfg.query_pool_mode,
                min_pool_size=cfg.query_n_select,
            )
        )

        if not ref_registered:
            reference_source = [np.asarray(last_result["pareto_F"], dtype=float)]
            if cfg.reference_covers_query_pool and len(query_f) > 0:
                reference_source.append(np.asarray(query_f, dtype=float))
            pareto_canonical = canonicalize_pareto_metrics_batch(
                np.vstack(reference_source)
            )
            if len(pareto_canonical) > 0:
                reference_rows = _pad_reference_rows(
                    pareto_canonical,
                    padding_ratio=cfg.reference_bounds_padding_ratio,
                )
                _apply_utility_event(
                    utility_learner,
                    shadow_utility_learner,
                    "set_reference",
                    reference_rows,
                )
                ref_registered = True
        inferred_global_generation = (
            int(cfg.n_gen_warmup)
            + int(iteration) * int(cfg.n_gen_per_interaction)
        )
        global_query_generation = (
            inferred_global_generation
            if cfg.consult_on_final_generation or query_pool_generation is None
            else int(query_pool_generation)
        )
        perform_learning_query(
            np.asarray(query_x, dtype=float),
            np.asarray(query_f, dtype=float),
            query_pool_source=str(query_pool_source),
            query_generation=global_query_generation,
            learning_iteration=len(x_history),
            phase=(
                "terminal"
                if iteration == cfg.rounds - 1 and cfg.consult_on_final_generation
                else "online"
            ),
            allow_search_side_effects=True,
        )


    assert last_result is not None
    if terminal_query_count:
        frozen_archive_x = np.asarray(last_result["archive_X"], dtype=float).copy()
        frozen_archive_f = np.asarray(last_result["archive_F"], dtype=float).copy()
        frozen_archive_paths = copy.deepcopy(last_result.get("archive_paths"))
        for _terminal_offset in range(terminal_query_count):
            perform_learning_query(
                frozen_archive_x,
                frozen_archive_f,
                query_pool_source="evaluation_archive",
                query_generation=int(cfg.total_gen_budget),
                learning_iteration=len(x_history),
                phase="terminal",
                allow_search_side_effects=False,
            )
        if not np.array_equal(
            frozen_archive_x,
            np.asarray(last_result["archive_X"], dtype=float),
        ):
            raise RuntimeError("terminal calibration mutated the frozen archive decisions")
        if not np.array_equal(
            frozen_archive_f,
            np.asarray(last_result["archive_F"], dtype=float),
        ):
            raise RuntimeError("terminal calibration mutated the frozen archive objectives")
        if frozen_archive_paths != last_result.get("archive_paths"):
            raise RuntimeError("terminal calibration mutated the frozen archive paths")

    final_shortlist: Optional[Dict[str, Any]] = None
    shadow_final_shortlist: Optional[Dict[str, Any]] = None
    shadow_member_42_archive_scores: List[float] = []
    utility_parameter_shadow_results: Dict[str, Dict[str, Any]] = {}
    final_shortlist_choice_idx: Optional[int] = None
    final_decision_reason: Optional[str] = None
    if int(cfg.final_shortlist_size) > 1:
        final_shortlist = _archive_shortlist_snapshot(
            last_result,
            utility_learner,
            shortlist_size=cfg.final_shortlist_size,
        )
        final_shown = canonicalize_pareto_metrics_batch(
            np.asarray(final_shortlist["metrics_raw"], dtype=float)
        ).tolist()
        if choice_provider is None:
            final_shortlist_choice_idx, final_decision_reason = choose_path(
                final_shown,
                true_weights,
                interactive,
                dm_mode="tchebycheff",
                ref_mins=(
                    None
                    if utility_learner._ref_min is None
                    else np.asarray(utility_learner._ref_min, dtype=float).tolist()
                ),
                ref_maxs=(
                    None
                    if utility_learner._ref_max is None
                    else np.asarray(utility_learner._ref_max, dtype=float).tolist()
                ),
                dm_noise_tau=cfg.dm_noise_tau,
                noise_kappa=cfg.noise_kappa,
            )
        else:
            final_request = _build_choice_request(
                phase="final_decision",
                iteration=len(x_history),
                selected_x=np.asarray(final_shortlist["archive_indices"], dtype=float)[:, None],
                selected_f=final_shortlist["metrics_raw"],
                source_x=np.asarray(final_shortlist["archive_indices"], dtype=float)[:, None],
                source_paths=final_shortlist["paths"],
                start=start,
                end=end,
                utility_learner=utility_learner,
                metadata={
                    "query_pool_source": "final_utility_archive_topk",
                    **_utility_choice_audit_metadata(
                        utility_learner,
                        final_shortlist["metrics_raw"],
                    ),
                },
            )
            final_shortlist_choice_idx, final_decision_reason = _choose_with_provider(
                choice_provider,
                final_request,
            )
        last_result = _apply_archive_shortlist_choice(
            last_result,
            final_shortlist,
            choice_idx=int(final_shortlist_choice_idx),
        )
    elif cfg.consult_on_final_generation:
        last_result = _reselect_archive_with_final_utility(
            last_result,
            utility_learner,
            use_evaluated_feasible=utility_objective_mode,
        )
    if shadow_utility_learner is not None:
        active_shadow_snapshot = _archive_shortlist_snapshot(
            last_result,
            utility_learner,
            shortlist_size=5,
        )
        from .slfn_shadow_bank import SLFNParameterShadowBank

        if isinstance(shadow_utility_learner, SLFNParameterShadowBank):
            utility_parameter_shadow_results = shadow_utility_learner.snapshot_archive(
                last_result["archive_F"],
                last_result["archive_paths"],
                shortlist_size=5,
            )
            baseline_shadow = utility_parameter_shadow_results.get("baseline")
            if baseline_shadow is None:
                raise RuntimeError("SLFN parameter shadow result lacks the baseline")
            if not np.array_equal(
                np.asarray(baseline_shadow["archive_learned_scores"], dtype=float),
                np.asarray(active_shadow_snapshot["archive_learned_scores"], dtype=float),
            ) or baseline_shadow["shortlist_archive_indices"] != active_shadow_snapshot[
                "archive_indices"
            ]:
                raise RuntimeError(
                    "SLFN parameter shadow baseline diverged from the active learner."
                )
        else:
            shadow_final_shortlist = _archive_shortlist_snapshot(
                last_result,
                shadow_utility_learner,
                shortlist_size=5,
            )
            shadow_member_42 = next(
                (
                    member
                    for member in (getattr(shadow_utility_learner, "members", None) or [])
                    if int(getattr(member, "rng_seed", -1)) == 42
                ),
                None,
            )
            if shadow_member_42 is None:
                shadow_member_42_archive_scores = list(
                    active_shadow_snapshot["archive_learned_scores"]
                )
            else:
                archive_f = np.asarray(last_result["archive_F"], dtype=float)
                shadow_member_42_archive_scores = [
                    float(score)
                    for score in np.asarray(
                        shadow_member_42.score_raw_batch(archive_f), dtype=float
                    ).reshape(-1)
                ]
                shadow_members = list(
                    getattr(shadow_utility_learner, "members", None) or []
                )
                seed42_position = shadow_members.index(shadow_member_42)
                seed42_receives_prior = seed42_position < int(
                    getattr(
                        shadow_utility_learner,
                        "prior_member_count",
                        len(shadow_members),
                    )
                )
                active_seed = int(
                    (utility_ensemble_provenance.get("active_member_seeds") or [42])[0]
                )
                if active_seed == 42 and seed42_receives_prior and not np.allclose(
                    shadow_member_42_archive_scores,
                    active_shadow_snapshot["archive_learned_scores"],
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise RuntimeError(
                        "Shadow ensemble seed-42 member diverged from the active seed-42 learner."
                    )
    final_metrics = canonicalize_best_metrics(last_result["best_metrics"])
    stored_best_path = last_result.get("best_path")
    if stored_best_path is not None:
        best_path = [tuple(point) for point in stored_best_path]
    else:
        try:
            best_path = _decode_candidate_path(last_result["best_vars"], start, end)
        except Exception:
            best_path = None
    acc, pairwise_loss, weight_l1_error, weight_cosine = _eval_session(
        weights,
        traces,
        x_history,
        y_history,
        true_weights,
        dm_noise_tau=cfg.dm_noise_tau,
        noise_kappa=cfg.noise_kappa,
    )
    if true_weights is not None and weight_l1_error is None:
        weight_l1_error = l1_error(weights, true_weights)
    learning_query_weights = [1.0] * len(traces)
    if weighted_terminal_top5 and learning_query_weights:
        learning_query_weights[-1] = 5.0
    elif geometric_terminal_top5 and learning_query_weights:
        geometric = np.power(
            temporal_weight_ratio,
            np.linspace(0.0, 1.0, len(learning_query_weights)),
        )
        geometric *= len(geometric) / float(np.sum(geometric))
        learning_query_weights = geometric.tolist()

    return SessionResult(
        method="proposed",
        final_weights=list(weights),
        true_weights=list(true_weights) if true_weights is not None else None,
        round_traces=traces,
        final_metrics=final_metrics,
        choice_reproduction_acc=acc,
        history_pairwise_loss=pairwise_loss,
        weight_l1_error=weight_l1_error,
        weight_cosine=weight_cosine,
        final_best_metrics_raw=list(last_result["best_metrics"]),
        final_best_vars=last_result["best_vars"],
        final_best_path=best_path,
        final_pareto_metrics_raw=np.asarray(last_result.get("pareto_F", [last_result["best_metrics"]]), dtype=float).tolist(),
        final_pareto_vars_raw=_final_pareto_vars_raw_list(last_result),
        final_archive_metrics_raw=(
            np.asarray(last_result["archive_F"], dtype=float).tolist()
            if last_result.get("archive_F") is not None
            else None
        ),
        final_archive_vars_raw=(
            np.asarray(last_result["archive_X"], dtype=float).tolist()
            if last_result.get("archive_X") is not None
            else None
        ),
        final_archive_paths=(
            [[tuple(point) for point in path] for path in last_result["archive_paths"]]
            if last_result.get("archive_paths") is not None
            else None
        ),
        final_evaluated_feasible_metrics_raw=(
            np.asarray(last_result["evaluated_feasible_F"], dtype=float).tolist()
            if last_result.get("evaluated_feasible_F") is not None
            else None
        ),
        evaluation_archive_stats=dict(last_result.get("archive_stats") or {}),
        recommendation_source=last_result.get("recommendation_source"),
        finite_run_nondominance_guaranteed=last_result.get("finite_run_nondominance_guaranteed"),
        final_utility_params=utility_learner.get_params(),
        final_structure_prior=copy.deepcopy(final_structure_prior),
        structure_trace=copy.deepcopy(structure_trace),
        structure_valid_rounds=list(structure_valid_rounds),
        query_pool_mode=str(cfg.query_pool_mode),
        query_selector_mode=str(cfg.query_selector_mode),
        query_pool_source=last_query_pool_source,
        query_pool_generation=last_query_pool_generation,
        utility_learner_variant=str(cfg.utility_learner_variant),
        utility_learner_kwargs=dict(cfg.utility_learner_kwargs or {}),
        utility_ensemble_mode=utility_ensemble_mode,
        utility_ensemble_provenance=copy.deepcopy(utility_ensemble_provenance),
        shadow_final_utility_params=(
            shadow_utility_learner.get_params()
            if shadow_utility_learner is not None
            and not utility_parameter_shadow_results
            else None
        ),
        shadow_final_archive_learned_scores=(
            list(shadow_final_shortlist["archive_learned_scores"])
            if shadow_final_shortlist is not None else []
        ),
        shadow_final_shortlist_archive_indices=(
            list(shadow_final_shortlist["archive_indices"])
            if shadow_final_shortlist is not None else []
        ),
        shadow_final_shortlist_metrics_raw=(
            copy.deepcopy(shadow_final_shortlist["metrics_raw"])
            if shadow_final_shortlist is not None else []
        ),
        shadow_final_member_42_archive_learned_scores=list(
            shadow_member_42_archive_scores
        ),
        utility_parameter_shadow_results=copy.deepcopy(
            utility_parameter_shadow_results
        ),
        fcpe_actual_variant=str(cfg.fcpe_actual_variant),
        preference_guidance_policy=str(cfg.preference_guidance_policy),
        history_prompt_mode=str(cfg.history_prompt_mode),
        environment_selection_policy=environment_selection_policy,
        post_feedback_selection_applied_rounds=list(
            post_feedback_selection_applied_rounds
        ),
        post_feedback_selection_changed_rounds=list(
            post_feedback_selection_changed_rounds
        ),
        final_shortlist_archive_indices=(
            list(final_shortlist["archive_indices"])
            if final_shortlist is not None else []
        ),
        final_shortlist_metrics_raw=(
            copy.deepcopy(final_shortlist["metrics_raw"])
            if final_shortlist is not None else []
        ),
        final_shortlist_paths=(
            copy.deepcopy(final_shortlist["paths"])
            if final_shortlist is not None else []
        ),
        final_archive_learned_scores=(
            list(final_shortlist["archive_learned_scores"])
            if final_shortlist is not None else []
        ),
        final_shortlist_choice_idx=final_shortlist_choice_idx,
        final_decision_reason=final_decision_reason,
        final_decision_count=1 if final_shortlist is not None else 0,
        learning_query_sources=list(learning_query_sources),
        learning_query_allocation=str(cfg.learning_query_allocation),
        learning_query_phases=list(learning_query_phases),
        learning_query_generations=list(learning_query_generations),
        learning_query_weights=list(learning_query_weights),
    )

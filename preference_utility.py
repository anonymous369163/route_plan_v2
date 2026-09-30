#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pairwise utility learner for the simplified anchored-residual mainline."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid


OBJECTIVES = ("Cost", "Length", "Safety", "Smoothness")
DIRECTIONS = ("min", "min", "max", "max")
_N_OBJ = len(OBJECTIVES)
PAIRWISE_UTILITY_VARIANTS = (
    "anchored_residual",
    "anchored_residual_relu",
    "benchmark_slfn",
    "benchmark_slfn_ensemble",
    "monotone_quadratic",
    "monotone_gated_mlp",
)


def _normalize_utility_variant(variant: Optional[str]) -> str:
    normalized = str(variant or "anchored_residual").strip().lower()
    if normalized == "scheme_a":
        return "monotone_quadratic"
    if normalized == "scheme_b":
        return "monotone_gated_mlp"
    return normalized


def _load_optional_torch_variants() -> Dict[str, Any]:
    module = importlib.import_module("preference_utility_torch_variants")
    return dict(getattr(module, "TORCH_UTILITY_VARIANT_CLASSES"))


def _reject_removed_utility_constructor_kwargs(kwargs: Dict[str, Any], *, caller: str) -> None:
    """Raise if callers pass kwargs that were removed from the API (fail loud, no silent ignore)."""
    if "min_prior_reg_factor" in kwargs:
        raise ValueError(
            f"{caller}: 'min_prior_reg_factor' is not supported (removed; it was never used). "
            "Remove it from utility_learner_kwargs, checkpoints, or other call sites."
        )


def build_pairwise_utility_learner(
    *,
    variant: str = "anchored_residual",
    **kwargs: Any,
):
    _reject_removed_utility_constructor_kwargs(kwargs, caller="build_pairwise_utility_learner")
    normalized = _normalize_utility_variant(variant)
    if normalized == "anchored_residual":
        return PairwiseUtilityLearner(**kwargs)
    if normalized == "anchored_residual_relu":
        return AnchoredResidualReLUUtilityLearner(**kwargs)
    if normalized == "benchmark_slfn":
        return BenchmarkSLFNUtilityLearner(**kwargs)
    if normalized == "benchmark_slfn_ensemble":
        return BenchmarkSLFNEnsembleUtilityLearner(**kwargs)
    if normalized not in PAIRWISE_UTILITY_VARIANTS:
        raise ValueError(
            "Unsupported utility learner variant: "
            f"{variant}. Supported variants are {PAIRWISE_UTILITY_VARIANTS}."
        )
    try:
        variant_classes = _load_optional_torch_variants()
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "Utility learner variant "
                f"'{normalized}' requires optional dependency 'torch'. "
                "Install PyTorch or use utility_learner_variant='anchored_residual'."
            ) from exc
        raise
    learner_cls = variant_classes.get(normalized)
    if learner_cls is None:
        raise ValueError(
            "Utility learner variant "
            f"'{normalized}' is not available in the optional torch variant module."
        )
    return learner_cls(**kwargs)


def _cross_pairs(n_obj: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n_obj) for j in range(i + 1, n_obj)]


def _to_preference_space(
    metrics_list: Sequence[Sequence[float]],
    ref_min: Optional[np.ndarray] = None,
    ref_max: Optional[np.ndarray] = None,
) -> np.ndarray:
    arr = np.asarray(metrics_list, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]

    lo = arr.min(axis=0) if ref_min is None else ref_min
    hi = arr.max(axis=0) if ref_max is None else ref_max
    normed = (arr - lo) / (hi - lo + 1e-8)
    normed = np.clip(normed, 0.0, 1.0)

    max_mask = np.array([direction == "max" for direction in DIRECTIONS])
    if np.any(max_mask):
        normed[:, max_mask] = 1.0 - normed[:, max_mask]
    return normed


def _raw_objectives_to_reference_space(
    metrics_list: Sequence[Sequence[float]],
    ref_min: Optional[np.ndarray] = None,
    ref_max: Optional[np.ndarray] = None,
) -> np.ndarray:
    arr = np.asarray(metrics_list, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] >= 4:
        readable = np.column_stack((arr[:, 0], arr[:, 1], -arr[:, 2], -arr[:, 3]))
    elif arr.shape[1] == 3:
        readable = np.column_stack((arr[:, 0], arr[:, 1], -arr[:, 2]))
    else:
        readable = arr.copy()
    return _to_preference_space(readable, ref_min=ref_min, ref_max=ref_max)


def _canonical_to_benchmark_min_space(metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(metrics_list, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    out = arr.copy()
    max_mask = np.array([direction == "max" for direction in DIRECTIONS[: out.shape[1]]], dtype=bool)
    if np.any(max_mask):
        out[:, max_mask] *= -1.0
    return out


def _benchmark_min_to_canonical_space(metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(metrics_list, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    out = arr.copy()
    max_mask = np.array([direction == "max" for direction in DIRECTIONS[: out.shape[1]]], dtype=bool)
    if np.any(max_mask):
        out[:, max_mask] *= -1.0
    return out


def _canonical_bounds_to_benchmark_bounds(
    ref_min: Sequence[float],
    ref_max: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(ref_min, dtype=np.float64).reshape(-1)
    hi = np.asarray(ref_max, dtype=np.float64).reshape(-1)
    raw_lo = lo.copy()
    raw_hi = hi.copy()
    max_mask = np.array([direction == "max" for direction in DIRECTIONS[: lo.shape[0]]], dtype=bool)
    if np.any(max_mask):
        raw_lo[max_mask] = -hi[max_mask]
        raw_hi[max_mask] = -lo[max_mask]
    return raw_lo, raw_hi


def _build_cross_features(features: np.ndarray) -> np.ndarray:
    if features.ndim == 1:
        features = features[None, :]
    pairs = [
        features[:, i] * features[:, j]
        for i in range(features.shape[1])
        for j in range(i + 1, features.shape[1])
    ]
    return np.column_stack(pairs) if pairs else np.zeros((features.shape[0], 0), dtype=float)


def _build_quadratic_features(features: np.ndarray) -> np.ndarray:
    """Full second-order features: diagonal (f_i^2) + cross (f_i * f_j)."""
    if features.ndim == 1:
        features = features[None, :]
    diag = features ** 2
    cross = _build_cross_features(features)
    return np.column_stack([diag, cross]) if cross.size > 0 else diag


class BenchmarkSLFNUtilityLearner:
    """Adapter that exposes benchmark FCPEUtilityLearner on the actual-problem API."""

    def __init__(
        self,
        n_obj: int = _N_OBJ,
        lambda_cross: float = 0.1,
        lambda_delta: float = 2.0,
        delta_cap: float = 0.75,
        min_rounds_for_residual: int = 2,
        interaction_loss: str = "pairwise_expand",
        n_hidden: int = 16,
        rng_seed: int = 42,
        use_full_quad: bool = True,
        early_cross_multiplier: float = 8.0,
        cross_unlock_round: int = 1,
        pair_prior_scale: float = 0.15,
        lambda_nl: Optional[float] = None,
        lambda_uninformed_cross: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _reject_removed_utility_constructor_kwargs(kwargs, caller="BenchmarkSLFNUtilityLearner.__init__")
        if kwargs:
            raise TypeError(
                f"BenchmarkSLFNUtilityLearner got unexpected keyword arguments: {sorted(kwargs)}"
            )
        from slfn_utility import FCPEUtilityLearner

        self.n_obj = int(n_obj)
        self.interaction_loss = str(interaction_loss)
        self.n_hidden = int(n_hidden)
        self.rng_seed = int(rng_seed)
        self.use_full_quad = bool(use_full_quad)
        self.lambda_cross = float(lambda_cross)
        self.lambda_nl = float(self.lambda_cross if lambda_nl is None else lambda_nl)
        if not np.isfinite(self.lambda_nl) or self.lambda_nl < 0.0:
            raise ValueError("lambda_nl must be a finite non-negative value")
        self.lambda_uninformed_cross = float(lambda_uninformed_cross)
        if not np.isfinite(self.lambda_uninformed_cross) or self.lambda_uninformed_cross < 0.0:
            raise ValueError("lambda_uninformed_cross must be a finite non-negative value")
        self.lambda_delta = float(lambda_delta)
        self.delta_cap = float(delta_cap)
        self.min_rounds_for_residual = int(min_rounds_for_residual)
        self.early_cross_multiplier = float(early_cross_multiplier)
        self.cross_unlock_round = int(cross_unlock_round)
        self.pair_prior_scale = float(max(0.0, pair_prior_scale))

        self._learner = FCPEUtilityLearner(
            n_obj=self.n_obj,
            lambda_cross=self.lambda_cross,
            lambda_delta=self.lambda_delta,
            delta_cap=self.delta_cap,
            min_rounds_for_residual=self.min_rounds_for_residual,
            interaction_loss=self.interaction_loss,
            n_hidden=self.n_hidden,
            rng_seed=self.rng_seed,
            use_full_quad=self.use_full_quad,
            early_cross_multiplier=self.early_cross_multiplier,
            cross_unlock_round=self.cross_unlock_round,
            lambda_nl=self.lambda_nl,
            lambda_uninformed_cross=self.lambda_uninformed_cross,
        )
        self.current_lambda_prior: float = self.lambda_delta
        self._ref_min: Optional[np.ndarray] = None
        self._ref_max: Optional[np.ndarray] = None

    def _canonical_to_model_space(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        return _canonical_to_benchmark_min_space(metrics_list)

    def _raw_to_model_space(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        # DSOP raw objectives already use the benchmark learner's all-min space:
        # [cost, length, -safety, -smoothness].  Flipping the last two columns
        # again would make runtime search inconsistent with canonical training.
        arr = np.asarray(metrics_list, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def set_reference(self, all_metrics: Sequence[Sequence[float]]) -> None:
        arr = np.asarray(all_metrics, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if len(arr) == 0:
            return
        self._ref_min = arr.min(axis=0)
        self._ref_max = arr.max(axis=0)
        self._learner.set_reference(self._canonical_to_model_space(arr))

    def set_reference_from_bounds(
        self,
        ref_min: Sequence[float],
        ref_max: Sequence[float],
    ) -> None:
        """Set normalisation envelope directly from known bounds.

        Constructs a two-row synthetic batch [lo, hi] so the inner FCPEUtilityLearner
        also receives a reference consistent with the provided bounds.
        """
        lo = np.asarray(ref_min, dtype=np.float64)
        hi = np.asarray(ref_max, dtype=np.float64)
        self._ref_min = lo
        self._ref_max = hi
        synthetic = np.vstack([lo[None, :], hi[None, :]])
        self._learner.set_reference(self._canonical_to_model_space(synthetic))

    def set_prior(self, weights: Sequence[float], confidence: float = 1.0) -> None:
        self._learner.set_base_weight_prior(weights, strength=float(confidence))
        self.current_lambda_prior = self.lambda_delta * float(np.clip(confidence, 0.0, 1.0))

    def set_structure_prior(
        self,
        pair_priors: Optional[Sequence[float]] = None,
        confidence: float = 0.0,
        pair_prior_scale: Optional[float] = None,
    ) -> None:
        scale = self.pair_prior_scale if pair_prior_scale is None else float(pair_prior_scale)
        self._learner.set_structure_prior(
            pair_priors=pair_priors,
            confidence=float(confidence),
            pair_prior_scale=scale,
        )

    def set_importance_order(
        self,
        order_pairs: Sequence[Tuple[int, int]],
        confidence: float = 0.5,
    ) -> None:
        self._learner.set_importance_order(order_pairs=order_pairs, confidence=float(confidence))

    def update(
        self,
        x_list: Sequence[Sequence[Sequence[float]]],
        y_list: Sequence[int],
        round_idx: int = 0,
        total_rounds: int = 3,
        round_weights: Optional[Sequence[float]] = None,
    ) -> None:
        _ = total_rounds
        rounds = [self._canonical_to_model_space(round_x) for round_x in x_list]
        self._learner.update(
            rounds,
            list(y_list),
            round_idx=int(round_idx),
            round_weights=round_weights,
        )

    def score(self, pref_features: Sequence[Sequence[float]]) -> np.ndarray:
        # The benchmark learner expects model-space/raw-space inputs here, not normalized preference features.
        return np.asarray(self._learner.score(pref_features), dtype=float).reshape(-1)

    def score_batch(self, pref_features: Sequence[Sequence[float]]) -> np.ndarray:
        return self.score(pref_features)

    def score_canonical_batch(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        model_space = self._canonical_to_model_space(metrics_list)
        return self.score(model_space)

    def score_raw_batch(
        self,
        raw_metrics: Sequence[Sequence[float]],
        is_infeasible: Optional[Sequence[bool]] = None,
    ) -> np.ndarray:
        model_space = self._raw_to_model_space(raw_metrics)
        scores = self.score(model_space)
        if is_infeasible is not None:
            infeasible = np.asarray(is_infeasible, dtype=bool).reshape(-1)
            if len(infeasible) == len(scores):
                scores = np.asarray(scores, dtype=float).copy()
                scores[infeasible] = 1e9
        return scores

    def get_linear_weights(self, min_weight: float = 0.01) -> List[float]:
        weights = np.asarray(self._learner.get_linear_weights(), dtype=float).reshape(-1)
        weights = np.clip(weights, 0.0, None)
        total = float(np.sum(weights))
        if total <= 0.0:
            weights = np.ones(self.n_obj, dtype=float) / float(self.n_obj)
        else:
            weights = weights / total
        weights = np.maximum(weights, float(min_weight))
        weights = weights / float(np.sum(weights))
        return weights.tolist()

    def has_active_cross_terms(self, threshold: float = 0.01) -> bool:
        cross_offset = self.n_obj if self.use_full_quad else 0
        cross_part = np.asarray(
            self._learner._a[cross_offset : cross_offset + self._learner.n_cross],
            dtype=float,
        )
        return bool(np.any(np.abs(cross_part) > float(threshold)))

    def count_trainable_parameters(self) -> int:
        return int(self.n_obj + self._learner.n_quad + self._learner.n_hidden)

    def to_preference_space(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        model_space = self._canonical_to_model_space(metrics_list)
        pref = self._learner._to_pref_space(model_space)
        return np.asarray(pref, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        return {
            "variant": "benchmark_slfn",
            "n_obj": self.n_obj,
            "interaction_loss": self.interaction_loss,
            "n_hidden": self.n_hidden,
            "rng_seed": self.rng_seed,
            "use_full_quad": self.use_full_quad,
            "lambda_cross": self.lambda_cross,
            "lambda_delta": self.lambda_delta,
            "delta_cap": self.delta_cap,
            "min_rounds_for_residual": self.min_rounds_for_residual,
            "early_cross_multiplier": self.early_cross_multiplier,
            "cross_unlock_round": self.cross_unlock_round,
            "pair_prior_scale": self.pair_prior_scale,
            "w_effective": self._learner._effective_w().tolist(),
            "base_w": self._learner._base_w.tolist(),
            "delta_w": self._learner._delta_w.tolist(),
            "a": self._learner._a.tolist(),
            "a_nl": self._learner._a_nl.tolist(),
            "W_nl": self._learner._W_nl.tolist(),
            "b_nl": self._learner._b_nl.tolist(),
            "lambda_nl": float(self.lambda_nl),
            "lambda_uninformed_cross": float(self.lambda_uninformed_cross),
            "structure_prior_target": self._learner._structure_prior_target.tolist(),
            "structure_confidence": float(self._learner._structure_confidence),
            "importance_pairs": [list(pair) for pair in self._learner._importance_pairs],
            "order_confidence": float(self._learner._order_confidence),
            "lambda_order": float(self._learner.lambda_order),
            "ref_min": None if self._ref_min is None else self._ref_min.tolist(),
            "ref_max": None if self._ref_max is None else self._ref_max.tolist(),
        }

    def load_params(self, params: Dict[str, Any]) -> "BenchmarkSLFNUtilityLearner":
        self._learner._base_w = np.asarray(
            params.get("base_w", np.ones(self.n_obj, dtype=float) / float(self.n_obj)),
            dtype=float,
        )
        self._learner._delta_w = np.asarray(
            params.get("delta_w", np.zeros(self.n_obj, dtype=float)),
            dtype=float,
        )
        self._learner._a = np.asarray(
            params.get("a", np.zeros(self._learner.n_quad, dtype=float)),
            dtype=float,
        )
        self._learner._a_nl = np.asarray(
            params.get("a_nl", np.zeros(self._learner.n_hidden, dtype=float)),
            dtype=float,
        )
        self._learner._W_nl = np.asarray(params.get("W_nl", self._learner._W_nl), dtype=float)
        self._learner._b_nl = np.asarray(params.get("b_nl", self._learner._b_nl), dtype=float)
        self.lambda_nl = float(params.get("lambda_nl", self.lambda_nl))
        self._learner.lambda_nl = self.lambda_nl
        self.lambda_uninformed_cross = float(
            params.get("lambda_uninformed_cross", self.lambda_uninformed_cross)
        )
        self._learner.lambda_uninformed_cross = self.lambda_uninformed_cross
        self._learner._structure_prior_target = np.asarray(
            params.get("structure_prior_target", np.zeros(self._learner.n_cross, dtype=float)),
            dtype=float,
        )
        self._learner._structure_confidence = float(
            np.clip(params.get("structure_confidence", 0.0), 0.0, 1.0)
        )
        self._learner._importance_pairs = [
            (int(pair[0]), int(pair[1])) for pair in params.get("importance_pairs", [])
        ]
        self._learner._order_confidence = float(
            np.clip(params.get("order_confidence", 0.0), 0.0, 1.0)
        )
        self._learner.lambda_order = float(params.get("lambda_order", self._learner.lambda_order))
        self.pair_prior_scale = float(
            max(0.0, params.get("pair_prior_scale", self.pair_prior_scale))
        )
        ref_min = params.get("ref_min")
        ref_max = params.get("ref_max")
        self._ref_min = None if ref_min is None else np.asarray(ref_min, dtype=float)
        self._ref_max = None if ref_max is None else np.asarray(ref_max, dtype=float)
        if self._ref_min is None or self._ref_max is None:
            self._learner._ref_min = None
            self._learner._ref_max = None
        else:
            raw_ref_min, raw_ref_max = _canonical_bounds_to_benchmark_bounds(
                self._ref_min,
                self._ref_max,
            )
            self._learner._ref_min = raw_ref_min
            self._learner._ref_max = raw_ref_max
        self.current_lambda_prior = float(self.lambda_delta)
        return self


class BenchmarkSLFNEnsembleUtilityLearner:
    """Arithmetic-score ensemble of independently seeded benchmark SLFN learners."""

    _DEFAULT_MEMBER_SEEDS = (42, 43, 44, 45, 46)

    def __init__(
        self,
        n_obj: int = _N_OBJ,
        lambda_cross: float = 0.1,
        lambda_delta: float = 2.0,
        delta_cap: float = 0.75,
        min_rounds_for_residual: int = 2,
        interaction_loss: str = "pairwise_expand",
        n_hidden: int = 16,
        rng_seed: int = 42,
        use_full_quad: bool = True,
        early_cross_multiplier: float = 8.0,
        cross_unlock_round: int = 1,
        pair_prior_scale: float = 0.15,
        member_seeds: Optional[Sequence[int]] = None,
        prior_member_count: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _reject_removed_utility_constructor_kwargs(
            kwargs, caller="BenchmarkSLFNEnsembleUtilityLearner.__init__"
        )
        if kwargs:
            raise TypeError(
                "BenchmarkSLFNEnsembleUtilityLearner got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )
        self.member_seeds = [
            int(seed)
            for seed in (
                self._DEFAULT_MEMBER_SEEDS
                if member_seeds is None and int(rng_seed) == 42
                else range(int(rng_seed), int(rng_seed) + 5)
                if member_seeds is None
                else member_seeds
            )
        ]
        if not self.member_seeds:
            raise ValueError("BenchmarkSLFNEnsembleUtilityLearner requires at least one member seed.")
        self.prior_member_count = (
            len(self.member_seeds)
            if prior_member_count is None
            else int(prior_member_count)
        )
        if not 0 <= self.prior_member_count <= len(self.member_seeds):
            raise ValueError("prior_member_count must be between zero and the member count")

        self.n_obj = int(n_obj)
        self.interaction_loss = str(interaction_loss)
        self.n_hidden = int(n_hidden)
        self.rng_seed = int(rng_seed)
        self.use_full_quad = bool(use_full_quad)
        self.lambda_cross = float(lambda_cross)
        self.lambda_delta = float(lambda_delta)
        self.delta_cap = float(delta_cap)
        self.min_rounds_for_residual = int(min_rounds_for_residual)
        self.early_cross_multiplier = float(early_cross_multiplier)
        self.cross_unlock_round = int(cross_unlock_round)
        self.pair_prior_scale = float(max(0.0, pair_prior_scale))
        self.members = [
            BenchmarkSLFNUtilityLearner(
                n_obj=self.n_obj,
                lambda_cross=self.lambda_cross,
                lambda_delta=self.lambda_delta,
                delta_cap=self.delta_cap,
                min_rounds_for_residual=self.min_rounds_for_residual,
                interaction_loss=self.interaction_loss,
                n_hidden=self.n_hidden,
                rng_seed=seed,
                use_full_quad=self.use_full_quad,
                early_cross_multiplier=self.early_cross_multiplier,
                cross_unlock_round=self.cross_unlock_round,
                pair_prior_scale=self.pair_prior_scale,
                # Behavior-only members have no structural target.  A zero-centred
                # ridge keeps their otherwise-unbounded cross terms identifiable.
                lambda_uninformed_cross=(
                    self.lambda_cross if member_index >= self.prior_member_count else 0.0
                ),
            )
            for member_index, seed in enumerate(self.member_seeds)
        ]
        self.current_lambda_prior = self.lambda_delta
        self._ref_min: Optional[np.ndarray] = None
        self._ref_max: Optional[np.ndarray] = None

    def _canonical_to_model_space(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        return self.members[0]._canonical_to_model_space(metrics_list)

    def _raw_to_model_space(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        return self.members[0]._raw_to_model_space(metrics_list)

    def set_reference(self, all_metrics: Sequence[Sequence[float]]) -> None:
        for member in self.members:
            member.set_reference(all_metrics)
        self._ref_min = self.members[0]._ref_min
        self._ref_max = self.members[0]._ref_max

    def set_reference_from_bounds(
        self,
        ref_min: Sequence[float],
        ref_max: Sequence[float],
    ) -> None:
        for member in self.members:
            member.set_reference_from_bounds(ref_min, ref_max)
        self._ref_min = self.members[0]._ref_min
        self._ref_max = self.members[0]._ref_max

    def set_prior(self, weights: Sequence[float], confidence: float = 1.0) -> None:
        for member in self.members:
            member.set_prior(weights, confidence=confidence)
        self.current_lambda_prior = self.members[0].current_lambda_prior

    def set_structure_prior(
        self,
        pair_priors: Optional[Sequence[float]] = None,
        confidence: float = 0.0,
        pair_prior_scale: Optional[float] = None,
    ) -> None:
        for member in self.members[: self.prior_member_count]:
            member.set_structure_prior(pair_priors, confidence, pair_prior_scale)

    def set_importance_order(
        self,
        order_pairs: Sequence[Tuple[int, int]],
        confidence: float = 0.5,
    ) -> None:
        for member in self.members[: self.prior_member_count]:
            member.set_importance_order(order_pairs, confidence=confidence)

    def update(
        self,
        x_list: Sequence[Sequence[Sequence[float]]],
        y_list: Sequence[int],
        round_idx: int = 0,
        total_rounds: int = 3,
        round_weights: Optional[Sequence[float]] = None,
    ) -> None:
        for member in self.members:
            member.update(
                x_list,
                y_list,
                round_idx=round_idx,
                total_rounds=total_rounds,
                round_weights=round_weights,
            )

    def score(self, pref_features: Sequence[Sequence[float]]) -> np.ndarray:
        scores = []
        for idx, member in enumerate(self.members):
            member_scores = np.asarray(member.score(pref_features), dtype=float).reshape(-1)
            if not np.all(np.isfinite(member_scores)):
                raise ValueError(f"SLFN ensemble member {idx} produced non-finite scores.")
            scores.append(member_scores)
        if any(score.shape != scores[0].shape for score in scores[1:]):
            raise ValueError("SLFN ensemble members produced score vectors with different shapes.")
        averaged = np.mean(np.vstack(scores), axis=0)
        if not np.all(np.isfinite(averaged)):
            raise ValueError("SLFN ensemble produced non-finite averaged scores.")
        return averaged

    def score_batch(self, pref_features: Sequence[Sequence[float]]) -> np.ndarray:
        return self.score(pref_features)

    def score_canonical_batch(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        return self.score(self._canonical_to_model_space(metrics_list))

    def score_raw_batch(
        self,
        raw_metrics: Sequence[Sequence[float]],
        is_infeasible: Optional[Sequence[bool]] = None,
    ) -> np.ndarray:
        scores = self.score(self._raw_to_model_space(raw_metrics))
        if is_infeasible is not None:
            infeasible = np.asarray(is_infeasible, dtype=bool).reshape(-1)
            if len(infeasible) == len(scores):
                scores = scores.copy()
                scores[infeasible] = 1e9
        return scores

    def get_linear_weights(self, min_weight: float = 0.01) -> List[float]:
        if len(self.members) == 1:
            return self.members[0].get_linear_weights(min_weight=min_weight)
        weights = np.mean(
            np.asarray([member.get_linear_weights(min_weight=0.0) for member in self.members]),
            axis=0,
        )
        weights = np.maximum(weights, float(min_weight))
        return (weights / float(np.sum(weights))).tolist()

    def has_active_cross_terms(self, threshold: float = 0.01) -> bool:
        return any(member.has_active_cross_terms(threshold=threshold) for member in self.members)

    def count_trainable_parameters(self) -> int:
        return sum(member.count_trainable_parameters() for member in self.members)

    def to_preference_space(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        return self.members[0].to_preference_space(metrics_list)

    def get_params(self) -> Dict[str, Any]:
        return {
            "variant": "benchmark_slfn_ensemble",
            "n_obj": self.n_obj,
            "member_seeds": list(self.member_seeds),
            "prior_member_count": int(self.prior_member_count),
            "members": [member.get_params() for member in self.members],
        }

    def load_params(self, params: Dict[str, Any]) -> "BenchmarkSLFNEnsembleUtilityLearner":
        prior_member_count = int(params.get("prior_member_count", self.prior_member_count))
        if not 0 <= prior_member_count <= len(self.members):
            raise ValueError("saved prior_member_count is outside the ensemble")
        self.prior_member_count = prior_member_count
        member_params = params.get("members")
        if not isinstance(member_params, list) or len(member_params) != len(self.members):
            raise ValueError("SLFN ensemble parameters must include one parameter mapping per member.")
        for member, saved_params in zip(self.members, member_params):
            member.load_params(saved_params)
        self._ref_min = self.members[0]._ref_min
        self._ref_max = self.members[0]._ref_max
        self.current_lambda_prior = self.members[0].current_lambda_prior
        return self


class PairwiseUtilityLearner:
    """Anchored residual utility learner with optional structure priors."""

    def __init__(
        self,
        n_obj: int = _N_OBJ,
        lambda_cross: float = 0.1,
        lambda_delta: Optional[float] = None,
        lambda_uniform: float = 0.5,
        early_cross_multiplier: float = 8.0,
        cross_unlock_round: int = 1,
        min_rounds_for_residual: int = 2,
        delta_cap: float = 0.75,
        pair_prior_scale: float = 0.15,
        lambda_order_base: float = 2.0,
        **kwargs: Any,
    ) -> None:
        _reject_removed_utility_constructor_kwargs(kwargs, caller="PairwiseUtilityLearner.__init__")
        if kwargs:
            raise TypeError(
                f"PairwiseUtilityLearner got unexpected keyword arguments: {sorted(kwargs)}"
            )
        self.n_obj = n_obj
        self.n_cross = self.n_obj * (self.n_obj - 1) // 2
        self.lambda_cross = float(lambda_cross)
        self.lambda_delta = float(lambda_delta if lambda_delta is not None else 2.0)
        # lambda_uniform: fallback stability regularization weight when LLM confidence is low.
        # Effective delta_reg = lambda_delta*conf + lambda_uniform*(1-conf), always nonzero.
        self.lambda_uniform = float(max(0.0, lambda_uniform))
        self.early_cross_multiplier = float(max(1.0, early_cross_multiplier))
        self.cross_unlock_round = int(max(0, cross_unlock_round))
        self.min_rounds_for_residual = int(max(1, min_rounds_for_residual))
        self.delta_cap = float(max(0.0, delta_cap))
        self.pair_prior_scale = float(max(0.0, pair_prior_scale))
        self.lambda_order_base = float(max(0.0, lambda_order_base))

        self._delta_w = np.zeros(self.n_obj, dtype=float)
        self._a = np.zeros(self.n_cross, dtype=float)
        self._w_llm: Optional[np.ndarray] = None
        self._confidence: float = 0.0
        self.current_lambda_prior: float = self.lambda_delta
        self.current_lambda_uniform: float = self.lambda_uniform
        self.current_lambda_cross: float = self.lambda_cross
        self._ref_min: Optional[np.ndarray] = None
        self._ref_max: Optional[np.ndarray] = None
        self._pair_prior = np.zeros(self.n_cross, dtype=float)
        self._pair_prior_target = np.zeros(self.n_cross, dtype=float)
        self._structure_confidence: float = 0.0
        self._current_pair_prior_scale: float = self.pair_prior_scale
        self._importance_pairs: List[Tuple[int, int]] = []
        self._order_confidence: float = 0.0

    def set_reference(self, all_metrics: Sequence[Sequence[float]]) -> None:
        arr = np.asarray(all_metrics, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        self._ref_min = arr.min(axis=0)
        self._ref_max = arr.max(axis=0)

    def set_reference_from_bounds(
        self,
        ref_min: Sequence[float],
        ref_max: Sequence[float],
    ) -> None:
        """Set normalisation envelope directly from known bounds.

        This is equivalent to ``set_reference`` but avoids the need to
        provide an actual metric batch — useful when the canonical reference
        bounds are pre-computed (e.g. from a PSI snapshot or a pooled archive)
        and must be shared across multiple runs for consistent scoring.
        """
        self._ref_min = np.asarray(ref_min, dtype=np.float64)
        self._ref_max = np.asarray(ref_max, dtype=np.float64)

    def _normalize_round(self, x_list: Sequence[Sequence[Sequence[float]]]) -> List[np.ndarray]:
        if self._ref_min is not None and self._ref_max is not None:
            return [_to_preference_space(round_x, self._ref_min, self._ref_max) for round_x in x_list]
        all_arr = np.vstack([np.asarray(round_x, dtype=np.float64) for round_x in x_list])
        return [_to_preference_space(round_x, all_arr.min(axis=0), all_arr.max(axis=0)) for round_x in x_list]

    def set_prior(self, w_llm: Sequence[float], confidence: float = 1.0) -> None:
        weights = np.asarray(w_llm, dtype=float)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        new_prior = weights / total if total > 0 else np.ones(self.n_obj, dtype=float) / self.n_obj

        # Remap _delta_w so effective weights stay consistent across prior updates.
        # Without this, swapping the LLM prior would silently shift w_eff = new_prior + old_delta,
        # which breaks the interpretation of what the optimizer has learned.
        if self._w_llm is not None and np.any(np.abs(self._delta_w) > 1e-12):
            old_eff = self._w_llm + self._delta_w
            remapped = old_eff - new_prior
            # Clamp to legal range for the new prior (w_eff[i] >= 1e-6, delta[i] <= delta_cap)
            remapped = np.clip(remapped, -new_prior + 1e-6, self.delta_cap)
            self._delta_w = remapped

        self._w_llm = new_prior
        self._confidence = float(np.clip(confidence, 0.0, 1.0))

    def set_structure_prior(
        self,
        pair_priors: Optional[Sequence[float]] = None,
        confidence: float = 0.0,
        pair_prior_scale: Optional[float] = None,
    ) -> None:
        pair = np.zeros(self.n_cross, dtype=float)
        if pair_priors is not None:
            arr = np.asarray(pair_priors, dtype=float).reshape(-1)
            if arr.size != self.n_cross:
                raise ValueError(f"pair_priors must have length {self.n_cross}, got {arr.size}")
            pair = np.clip(arr, 0.0, None)
        self._pair_prior = pair
        self._structure_confidence = float(np.clip(confidence, 0.0, 1.0))
        self._current_pair_prior_scale = float(
            self.pair_prior_scale if pair_prior_scale is None else max(0.0, pair_prior_scale)
        )
        self._pair_prior_target = self._current_pair_prior_scale * self._pair_prior
        if not np.any(np.abs(self._a) > 1e-12):
            self._a = self._pair_prior_target.copy()

    def set_importance_order(
        self,
        order_pairs: Sequence[Tuple[int, int]],
        confidence: float = 0.5,
    ) -> None:
        """Set soft ordering constraints: each (higher_idx, lower_idx) means w[higher] > w[lower]."""
        self._importance_pairs = [(int(h), int(l)) for h, l in order_pairs]
        self._order_confidence = float(np.clip(confidence, 0.0, 1.0))

    def _current_prior(self) -> np.ndarray:
        if self._w_llm is None:
            return np.ones(self.n_obj, dtype=float) / self.n_obj
        return self._w_llm

    def _effective_linear(self, delta_w: Optional[np.ndarray] = None) -> np.ndarray:
        return self._current_prior() + np.asarray(self._delta_w if delta_w is None else delta_w, dtype=float)

    def update(
        self,
        x_list: Sequence[Sequence[Sequence[float]]],
        y_list: Sequence[int],
        round_idx: int = 0,
        total_rounds: int = 3,
    ) -> None:
        if len(x_list) < self.min_rounds_for_residual:
            self._delta_w = np.zeros(self.n_obj, dtype=float)
            self._a = self._pair_prior_target.copy()
            self.current_lambda_prior = self.lambda_delta
            self.current_lambda_cross = self.lambda_cross * self.early_cross_multiplier
            return

        pref_rounds = self._normalize_round(x_list)
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for pref_round, chosen_idx in zip(pref_rounds, y_list):
            chosen = pref_round[chosen_idx]
            for idx in range(len(pref_round)):
                if idx != chosen_idx:
                    pairs.append((chosen, pref_round[idx]))
        if not pairs:
            return

        chosen_arr = np.stack([left for left, _ in pairs])
        rejected_arr = np.stack([right for _, right in pairs])
        diff_linear = rejected_arr - chosen_arr
        diff_cross = _build_cross_features(rejected_arr) - _build_cross_features(chosen_arr)

        confidence_clipped = float(np.clip(self._confidence, 0.0, 1.0))
        self.current_lambda_prior = self.lambda_delta * confidence_clipped
        self.current_lambda_uniform = self.lambda_uniform * (1.0 - confidence_clipped)
        cross_scale = self.early_cross_multiplier if round_idx < self.cross_unlock_round else 1.0
        if len(x_list) <= 1:
            cross_scale *= 2.0
        self.current_lambda_cross = self.lambda_cross * cross_scale

        prior_w = self._current_prior()
        theta0 = np.concatenate([self._delta_w, self._a])
        pair_prior_target = self._pair_prior_target
        bounds = [
            (float(-prior_w[idx] + 1e-6), float(self.delta_cap))
            for idx in range(self.n_obj)
        ] + [(None, None)] * self.n_cross

        importance_pairs = self._importance_pairs
        structure_confidence = self._structure_confidence
        lambda_order = self.lambda_order_base * self._order_confidence
        order_margin = 0.02

        lambda_delta_eff = self.current_lambda_prior + self.current_lambda_uniform

        def objective(theta: np.ndarray) -> float:
            delta = theta[: self.n_obj]
            cross = theta[self.n_obj :]
            w = prior_w + delta
            logits = diff_linear @ w + diff_cross @ cross
            rank_loss = -np.sum(np.log(sigmoid(logits) + 1e-12))
            delta_reg = lambda_delta_eff * np.sum(delta ** 2)
            cross_reg = (
                self.current_lambda_cross
                * structure_confidence
                * np.sum((cross - pair_prior_target) ** 2)
            )
            order_pen = 0.0
            for hi, lo in importance_pairs:
                violation = w[lo] - w[hi] + order_margin
                if violation > 0.0:
                    order_pen += violation * violation
            return float(rank_loss + delta_reg + cross_reg + lambda_order * order_pen)

        def gradient(theta: np.ndarray) -> np.ndarray:
            delta = theta[: self.n_obj]
            cross = theta[self.n_obj :]
            w = prior_w + delta
            logits = diff_linear @ w + diff_cross @ cross
            residual = 1.0 - sigmoid(logits)
            grad_delta = -(diff_linear.T @ residual) + 2.0 * lambda_delta_eff * delta
            grad_cross = (
                -(diff_cross.T @ residual)
                + 2.0 * self.current_lambda_cross * structure_confidence * (cross - pair_prior_target)
            )
            for hi, lo in importance_pairs:
                violation = w[lo] - w[hi] + order_margin
                if violation > 0.0:
                    grad_delta[lo] += 2.0 * lambda_order * violation
                    grad_delta[hi] -= 2.0 * lambda_order * violation
            return np.concatenate([grad_delta, grad_cross])

        result = minimize(
            objective,
            theta0,
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        self._delta_w = result.x[: self.n_obj].copy()
        self._a = result.x[self.n_obj :].copy()

    def _compute(self, pref_features: np.ndarray) -> np.ndarray:
        if pref_features.ndim == 1:
            pref_features = pref_features[None, :]
        linear = pref_features @ self._effective_linear()
        cross = _build_cross_features(pref_features) @ self._a
        return linear + cross

    def score(self, pref_features: Sequence[float]) -> float:
        arr = np.asarray(pref_features, dtype=float)
        return float(self._compute(arr.reshape(1, -1))[0])

    def score_batch(self, pref_features: np.ndarray) -> np.ndarray:
        arr = np.asarray(pref_features, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        return self._compute(arr)

    def score_canonical_batch(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        pref = _to_preference_space(metrics_list, ref_min=self._ref_min, ref_max=self._ref_max)
        return self._compute(pref)

    def score_raw_batch(
        self,
        raw_metrics: Sequence[Sequence[float]],
        is_infeasible: Optional[Sequence[bool]] = None,
    ) -> np.ndarray:
        pref = _raw_objectives_to_reference_space(raw_metrics, ref_min=self._ref_min, ref_max=self._ref_max)
        scores = self._compute(pref)
        if is_infeasible is not None:
            infeasible = np.asarray(is_infeasible, dtype=bool).reshape(-1)
            if len(infeasible) == len(scores):
                scores = np.asarray(scores, dtype=float).copy()
                scores[infeasible] = 1e9
        return scores

    def get_linear_weights(self, min_weight: float = 0.01) -> List[float]:
        weights = np.clip(self._effective_linear(), 0.0, None)
        total = float(weights.sum())
        weights = weights / total if total > 0 else np.ones(self.n_obj, dtype=float) / self.n_obj
        weights = np.maximum(weights, min_weight)
        weights = weights / float(weights.sum())
        return weights.tolist()

    def get_params(self) -> Dict[str, Any]:
        return {
            "w_effective": self._effective_linear().tolist(),
            "delta_w": self._delta_w.tolist(),
            "a": self._a.tolist(),
            "lambda_prior": self.current_lambda_prior,
            "lambda_uniform": self.current_lambda_uniform,
            "lambda_cross": self.current_lambda_cross,
            "confidence": self._confidence,
            "structure_confidence": self._structure_confidence,
            "w_llm": self._current_prior().tolist(),
            "ref_min": self._ref_min.tolist() if self._ref_min is not None else None,
            "ref_max": self._ref_max.tolist() if self._ref_max is not None else None,
            "pair_priors": self._pair_prior.tolist(),
            "pair_prior_target": self._pair_prior_target.tolist(),
            "pair_prior_scale": self.pair_prior_scale,
            "current_pair_prior_scale": self._current_pair_prior_scale,
            "cross_pairs": _cross_pairs(self.n_obj),
            "importance_pairs": self._importance_pairs,
            "order_confidence": self._order_confidence,
            "lambda_order_base": self.lambda_order_base,
            "lambda_order": self.lambda_order_base * self._order_confidence,
            "variant": "anchored_residual",
        }

    def has_active_cross_terms(self, threshold: float = 0.01) -> bool:
        return bool(np.any(np.abs(self._a) > threshold))

    def count_trainable_parameters(self) -> int:
        return int(self.n_obj + self.n_cross)

    def load_params(self, params: Dict[str, Any]) -> "PairwiseUtilityLearner":
        self._delta_w = np.asarray(params.get("delta_w", np.zeros(self.n_obj)), dtype=float)
        self._a = np.asarray(params.get("a", np.zeros(self.n_cross)), dtype=float)
        self._w_llm = np.asarray(params.get("w_llm", np.ones(self.n_obj) / self.n_obj), dtype=float)
        self._confidence = float(np.clip(params.get("confidence", 0.0), 0.0, 1.0))
        self.current_lambda_prior = float(params.get("lambda_prior", self.lambda_delta))
        self.current_lambda_cross = float(params.get("lambda_cross", self.lambda_cross))
        self.current_lambda_uniform = float(params.get("lambda_uniform", self.lambda_uniform))
        self._structure_confidence = float(np.clip(params.get("structure_confidence", 0.0), 0.0, 1.0))
        ref_min = params.get("ref_min")
        ref_max = params.get("ref_max")
        self._ref_min = None if ref_min is None else np.asarray(ref_min, dtype=float)
        self._ref_max = None if ref_max is None else np.asarray(ref_max, dtype=float)
        self._pair_prior = np.asarray(params.get("pair_priors", np.zeros(self.n_cross)), dtype=float)
        self._pair_prior_target = np.asarray(
            params.get("pair_prior_target", np.zeros(self.n_cross)),
            dtype=float,
        )
        self.pair_prior_scale = float(params.get("pair_prior_scale", self.pair_prior_scale))
        self._current_pair_prior_scale = float(
            params.get("current_pair_prior_scale", self.pair_prior_scale)
        )
        self._importance_pairs = [tuple(p) for p in params.get("importance_pairs", [])]
        self.lambda_order_base = float(params.get("lambda_order_base", self.lambda_order_base))
        self._order_confidence = float(np.clip(params.get("order_confidence", 0.0), 0.0, 1.0))
        if not np.any(np.abs(self._pair_prior_target) > 1e-12):
            self._pair_prior_target = self._current_pair_prior_scale * self._pair_prior
        return self

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "PairwiseUtilityLearner":
        variant = _normalize_utility_variant(params.get("variant", "anchored_residual"))
        n_obj = int(params.get("n_obj", len(params.get("w_effective") or params.get("w_llm") or OBJECTIVES)))
        if variant == "anchored_residual":
            learner = cls(n_obj=n_obj)
            learner.load_params(params)
            return learner
        if variant == "anchored_residual_relu":
            n_hidden = int(params.get("n_hidden", 2 * n_obj))
            rng_seed = int(params.get("rng_seed", 42))
            learner = AnchoredResidualReLUUtilityLearner(
                n_obj=n_obj, n_hidden=n_hidden, rng_seed=rng_seed,
            )
            learner.load_params(params)
            return learner
        if variant == "benchmark_slfn":
            learner = BenchmarkSLFNUtilityLearner(
                n_obj=n_obj,
                interaction_loss=str(params.get("interaction_loss", "pairwise_expand")),
                n_hidden=int(params.get("n_hidden", 16)),
                rng_seed=int(params.get("rng_seed", 42)),
                use_full_quad=bool(params.get("use_full_quad", True)),
                lambda_cross=float(params.get("lambda_cross", 0.1)),
                lambda_delta=float(params.get("lambda_delta", 2.0)),
                delta_cap=float(params.get("delta_cap", 0.75)),
                min_rounds_for_residual=int(params.get("min_rounds_for_residual", 2)),
                early_cross_multiplier=float(params.get("early_cross_multiplier", 8.0)),
                cross_unlock_round=int(params.get("cross_unlock_round", 1)),
                pair_prior_scale=float(params.get("pair_prior_scale", 0.15)),
            )
            learner.load_params(params)
            return learner
        if variant == "benchmark_slfn_ensemble":
            member_params = params.get("members")
            member_seeds = params.get("member_seeds")
            if not isinstance(member_params, list) or not isinstance(member_seeds, list):
                raise ValueError("SLFN ensemble parameters must include member_seeds and members.")
            first_member = member_params[0] if member_params else {}
            learner = BenchmarkSLFNEnsembleUtilityLearner(
                n_obj=n_obj,
                interaction_loss=str(first_member.get("interaction_loss", "pairwise_expand")),
                n_hidden=int(first_member.get("n_hidden", 16)),
                use_full_quad=bool(first_member.get("use_full_quad", True)),
                lambda_cross=float(first_member.get("lambda_cross", 0.1)),
                lambda_delta=float(first_member.get("lambda_delta", 2.0)),
                delta_cap=float(first_member.get("delta_cap", 0.75)),
                min_rounds_for_residual=int(first_member.get("min_rounds_for_residual", 2)),
                early_cross_multiplier=float(first_member.get("early_cross_multiplier", 8.0)),
                cross_unlock_round=int(first_member.get("cross_unlock_round", 1)),
                pair_prior_scale=float(first_member.get("pair_prior_scale", 0.15)),
                member_seeds=member_seeds,
            )
            learner.load_params(params)
            return learner
        learner = build_pairwise_utility_learner(variant=variant, n_obj=n_obj)
        learner.load_params(params)
        return learner

    def to_preference_space(self, metrics_list: Sequence[Sequence[float]]) -> np.ndarray:
        return _to_preference_space(metrics_list, ref_min=self._ref_min, ref_max=self._ref_max)


class AnchoredResidualReLUUtilityLearner(PairwiseUtilityLearner):
    """Anchored residual + full quadratic features + fixed random ReLU hidden layer.

    Scoring formula:
        score(pref) = pref @ w_eff + quad_features(pref) @ a_quad + ReLU(pref @ W + b) @ a_nl

    The random projection weights (W, b) are fixed at construction; only the output
    coefficients (a_quad, a_nl) are learned alongside the linear delta_w and cross terms.
    """

    def __init__(
        self,
        n_obj: int = _N_OBJ,
        n_hidden: Optional[int] = None,
        rng_seed: int = 42,
        lambda_nl: float = 0.1,
        use_full_quad: bool = True,
        **kwargs: Any,
    ) -> None:
        # Pop our extra kwargs before passing to parent (parent rejects unknown kwargs)
        super().__init__(n_obj=n_obj, **kwargs)

        self.use_full_quad = bool(use_full_quad)
        self.n_quad = (n_obj + self.n_cross) if self.use_full_quad else self.n_cross
        self.n_hidden = int(n_hidden if n_hidden is not None else 2 * n_obj)
        self.rng_seed = int(rng_seed)
        self.lambda_nl = float(lambda_nl)

        rng = np.random.RandomState(self.rng_seed)
        self._W_nl = rng.standard_normal((n_obj, self.n_hidden)) / np.sqrt(n_obj)
        self._b_nl = rng.uniform(-1.0, 1.0, size=self.n_hidden)

        self._a_quad = np.zeros(self.n_quad, dtype=float)
        self._a_nl = np.zeros(self.n_hidden, dtype=float)

    def _build_quad(self, pref: np.ndarray) -> np.ndarray:
        if self.use_full_quad:
            return _build_quadratic_features(pref)
        return _build_cross_features(pref)

    def _build_nl_features(self, pref: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.atleast_2d(pref) @ self._W_nl + self._b_nl)

    def set_structure_prior(
        self,
        pair_priors: Optional[Sequence[float]] = None,
        confidence: float = 0.0,
        pair_prior_scale: Optional[float] = None,
    ) -> None:
        super().set_structure_prior(pair_priors, confidence, pair_prior_scale)
        cross_offset = self.n_obj if self.use_full_quad else 0
        if not np.any(np.abs(self._a_quad[cross_offset:cross_offset + self.n_cross]) > 1e-12):
            self._a_quad[cross_offset:cross_offset + self.n_cross] = self._pair_prior_target.copy()

    def update(
        self,
        x_list: Sequence[Sequence[Sequence[float]]],
        y_list: Sequence[int],
        round_idx: int = 0,
        total_rounds: int = 3,
    ) -> None:
        n_obj = self.n_obj
        n_quad = self.n_quad
        n_hidden = self.n_hidden
        n_cross = self.n_cross
        cross_offset = n_obj if self.use_full_quad else 0

        if len(x_list) < self.min_rounds_for_residual:
            self._delta_w = np.zeros(n_obj, dtype=float)
            self._a = self._pair_prior_target.copy()
            self._a_quad = np.zeros(n_quad, dtype=float)
            self._a_quad[cross_offset:cross_offset + n_cross] = self._pair_prior_target.copy()
            self._a_nl = np.zeros(n_hidden, dtype=float)
            self.current_lambda_prior = self.lambda_delta
            self.current_lambda_cross = self.lambda_cross * self.early_cross_multiplier
            return

        pref_rounds = self._normalize_round(x_list)
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for pref_round, chosen_idx in zip(pref_rounds, y_list):
            chosen = pref_round[chosen_idx]
            for idx in range(len(pref_round)):
                if idx != chosen_idx:
                    pairs.append((chosen, pref_round[idx]))
        if not pairs:
            return

        chosen_arr = np.stack([left for left, _ in pairs])
        rejected_arr = np.stack([right for _, right in pairs])
        diff_linear = rejected_arr - chosen_arr
        diff_quad = self._build_quad(rejected_arr) - self._build_quad(chosen_arr)
        diff_nl = self._build_nl_features(rejected_arr) - self._build_nl_features(chosen_arr)

        confidence_clipped = float(np.clip(self._confidence, 0.0, 1.0))
        self.current_lambda_prior = self.lambda_delta * confidence_clipped
        self.current_lambda_uniform = self.lambda_uniform * (1.0 - confidence_clipped)
        cross_scale = self.early_cross_multiplier if round_idx < self.cross_unlock_round else 1.0
        if len(x_list) <= 1:
            cross_scale *= 2.0
        self.current_lambda_cross = self.lambda_cross * cross_scale

        prior_w = self._current_prior()
        theta0 = np.concatenate([self._delta_w, self._a_quad, self._a_nl])
        pair_prior_target = self._pair_prior_target
        bounds = (
            [(float(-prior_w[i] + 1e-6), float(self.delta_cap)) for i in range(n_obj)]
            + [(None, None)] * n_quad
            + [(None, None)] * n_hidden
        )

        importance_pairs = self._importance_pairs
        structure_confidence = self._structure_confidence
        lambda_order = self.lambda_order_base * self._order_confidence
        order_margin = 0.02
        lambda_delta_eff = self.current_lambda_prior + self.current_lambda_uniform
        lam_c = self.current_lambda_cross
        lam_nl = self.lambda_nl

        def objective(theta: np.ndarray) -> float:
            delta = theta[:n_obj]
            q = theta[n_obj:n_obj + n_quad]
            a_nl = theta[n_obj + n_quad:]
            w = prior_w + delta
            logits = diff_linear @ w + diff_quad @ q + diff_nl @ a_nl
            rank_loss = -np.sum(np.log(sigmoid(logits) + 1e-12))

            q_diag = q[:cross_offset]
            q_cross = q[cross_offset:cross_offset + n_cross]
            diff_cross = q_cross - pair_prior_target

            reg = float(
                lambda_delta_eff * np.sum(delta ** 2)
                + lam_c * np.sum(q_diag ** 2)
                + lam_c * structure_confidence * np.sum(diff_cross ** 2)
                + lam_nl * np.sum(a_nl ** 2)
            )
            order_pen = 0.0
            for hi, lo in importance_pairs:
                violation = w[lo] - w[hi] + order_margin
                if violation > 0.0:
                    order_pen += violation * violation
            return float(rank_loss + reg + lambda_order * order_pen)

        def gradient(theta: np.ndarray) -> np.ndarray:
            delta = theta[:n_obj]
            q = theta[n_obj:n_obj + n_quad]
            a_nl = theta[n_obj + n_quad:]
            w = prior_w + delta
            logits = diff_linear @ w + diff_quad @ q + diff_nl @ a_nl
            resid = 1.0 - sigmoid(logits)

            q_diag = q[:cross_offset]
            q_cross = q[cross_offset:cross_offset + n_cross]
            diff_cross = q_cross - pair_prior_target

            g_d = -(diff_linear.T @ resid) + 2.0 * lambda_delta_eff * delta
            g_q = np.zeros(n_quad, dtype=float)
            g_q[:cross_offset] = 2.0 * lam_c * q_diag
            g_q[cross_offset:cross_offset + n_cross] = 2.0 * lam_c * structure_confidence * diff_cross
            g_q -= diff_quad.T @ resid
            g_nl = -(diff_nl.T @ resid) + 2.0 * lam_nl * a_nl

            for hi, lo in importance_pairs:
                violation = w[lo] - w[hi] + order_margin
                if violation > 0.0:
                    g_d[lo] += 2.0 * lambda_order * violation
                    g_d[hi] -= 2.0 * lambda_order * violation
            return np.concatenate([g_d, g_q, g_nl])

        result = minimize(
            objective,
            theta0,
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        self._delta_w = result.x[:n_obj].copy()
        self._a_quad = result.x[n_obj:n_obj + n_quad].copy()
        self._a_nl = result.x[n_obj + n_quad:].copy()
        cross_offset_s = self.n_obj if self.use_full_quad else 0
        self._a = self._a_quad[cross_offset_s:cross_offset_s + self.n_cross].copy()

    def _compute(self, pref_features: np.ndarray) -> np.ndarray:
        if pref_features.ndim == 1:
            pref_features = pref_features[None, :]
        linear = pref_features @ self._effective_linear()
        quad = self._build_quad(pref_features) @ self._a_quad
        nl = self._build_nl_features(pref_features) @ self._a_nl
        return linear + quad + nl

    def get_params(self) -> Dict[str, Any]:
        base = super().get_params()
        base["variant"] = "anchored_residual_relu"
        base["n_hidden"] = self.n_hidden
        base["rng_seed"] = self.rng_seed
        base["use_full_quad"] = self.use_full_quad
        base["lambda_nl"] = self.lambda_nl
        base["a_quad"] = self._a_quad.tolist()
        base["a_nl"] = self._a_nl.tolist()
        base["W_nl"] = self._W_nl.tolist()
        base["b_nl"] = self._b_nl.tolist()
        return base

    def load_params(self, params: Dict[str, Any]) -> "AnchoredResidualReLUUtilityLearner":
        super().load_params(params)
        if "a_quad" in params:
            self._a_quad = np.asarray(params["a_quad"], dtype=float)
        else:
            cross_offset = self.n_obj if self.use_full_quad else 0
            self._a_quad = np.zeros(self.n_quad, dtype=float)
            self._a_quad[cross_offset:cross_offset + self.n_cross] = self._a.copy()
        if "a_nl" in params:
            self._a_nl = np.asarray(params["a_nl"], dtype=float)
        if "W_nl" in params:
            self._W_nl = np.asarray(params["W_nl"], dtype=float)
        if "b_nl" in params:
            self._b_nl = np.asarray(params["b_nl"], dtype=float)
        return self

    def count_trainable_parameters(self) -> int:
        return int(self.n_obj + self.n_quad + self.n_hidden)

    def has_active_cross_terms(self, threshold: float = 0.01) -> bool:
        cross_offset = self.n_obj if self.use_full_quad else 0
        cross_part = self._a_quad[cross_offset:cross_offset + self.n_cross]
        return bool(np.any(np.abs(cross_part) > threshold))

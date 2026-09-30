"""Core dataclasses for the simplified preference-agent mainline."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmark_defaults import FORMAL_PROPOSED_SESSION_KWARGS
from config import DEFAULT_LLM_MODEL
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _default_session_guidance_ratio() -> float:
    return float(FORMAL_PROPOSED_SESSION_KWARGS["guidance_ratio"])


def _default_session_restart_bias() -> float:
    return float(FORMAL_PROPOSED_SESSION_KWARGS["preference_restart_bias_ratio"])


@dataclass
class RoundTrace:
    iteration: int
    presented_metrics: List[List[float]]
    choice_idx: int
    user_reason: str
    method_weights_before: List[float]
    method_weights_after: List[float]
    focus: List[str]
    verifier_scores: List[Dict[str, Any]] = field(default_factory=list)
    utility_params: Optional[Dict[str, Any]] = None
    agg_prior_weights: Optional[List[float]] = None
    posterior_confidence: Optional[float] = None
    prior_strategy: Optional[str] = None
    feedback_used: Optional[bool] = None
    lambda_prior_used: Optional[float] = None
    cross_term_active: Optional[bool] = None
    structure_prior: Optional[Dict[str, Any]] = None
    dm_ref_mins: Optional[List[float]] = None
    dm_ref_maxs: Optional[List[float]] = None


@dataclass
class SessionResult:
    method: str
    final_weights: List[float]
    true_weights: Optional[List[float]]
    round_traces: List[RoundTrace]
    final_metrics: Optional[List[float]]
    choice_reproduction_acc: float
    history_pairwise_loss: float
    weight_l1_error: Optional[float]
    weight_cosine: Optional[float]
    final_best_metrics_raw: Optional[List[float]] = None
    # Lossy fixed-width compatibility projection; final_best_path is exact.
    final_best_vars: Optional[np.ndarray] = None
    final_best_path: Optional[List[Tuple[int, int]]] = None
    final_pareto_metrics_raw: Optional[List[List[float]]] = None
    final_pareto_vars_raw: Optional[List[List[float]]] = None
    final_archive_metrics_raw: Optional[List[List[float]]] = None
    # Lossy fixed-width compatibility projections; final_archive_paths is exact.
    final_archive_vars_raw: Optional[List[List[float]]] = None
    final_archive_paths: Optional[List[List[Tuple[float, float]]]] = None
    final_evaluated_feasible_metrics_raw: Optional[List[List[float]]] = None
    evaluation_archive_stats: Dict[str, Any] = field(default_factory=dict)
    recommendation_source: Optional[str] = None
    finite_run_nondominance_guaranteed: Optional[bool] = None
    final_utility_params: Optional[Dict[str, Any]] = None
    final_structure_prior: Optional[Dict[str, Any]] = None
    structure_trace: List[Dict[str, Any]] = field(default_factory=list)
    structure_valid_rounds: List[int] = field(default_factory=list)
    query_pool_mode: Optional[str] = None
    query_selector_mode: Optional[str] = None
    query_pool_source: Optional[str] = None
    query_pool_generation: Optional[int] = None
    utility_learner_variant: Optional[str] = None
    utility_learner_kwargs: Dict[str, Any] = field(default_factory=dict)
    utility_ensemble_mode: str = "off"
    utility_ensemble_provenance: Dict[str, Any] = field(default_factory=dict)
    shadow_final_utility_params: Optional[Dict[str, Any]] = None
    shadow_final_archive_learned_scores: List[float] = field(default_factory=list)
    shadow_final_shortlist_archive_indices: List[int] = field(default_factory=list)
    shadow_final_shortlist_metrics_raw: List[List[float]] = field(default_factory=list)
    shadow_final_member_42_archive_learned_scores: List[float] = field(default_factory=list)
    utility_parameter_shadow_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fcpe_actual_variant: str = "default"
    preference_guidance_policy: str = "current_full_search"
    history_prompt_mode: str = "current"
    environment_selection_policy: Optional[str] = None
    post_feedback_selection_applied_rounds: List[int] = field(default_factory=list)
    post_feedback_selection_changed_rounds: List[int] = field(default_factory=list)
    final_shortlist_archive_indices: List[int] = field(default_factory=list)
    final_shortlist_metrics_raw: List[List[float]] = field(default_factory=list)
    final_shortlist_paths: List[List[Tuple[float, float]]] = field(default_factory=list)
    final_archive_learned_scores: List[float] = field(default_factory=list)
    final_shortlist_choice_idx: Optional[int] = None
    final_decision_reason: Optional[str] = None
    final_decision_count: int = 0
    learning_query_sources: List[str] = field(default_factory=list)
    learning_query_allocation: str = "paper_11_plus_1"
    learning_query_phases: List[str] = field(default_factory=list)
    learning_query_generations: List[int] = field(default_factory=list)
    learning_query_weights: List[float] = field(default_factory=list)


@dataclass
class SessionConfig:
    rounds: int = 4
    llm_model: str = DEFAULT_LLM_MODEL
    llm_temperature: float = 0.0
    query_n_select: int = 5
    n_gen_warmup: int = 20
    n_gen_per_interaction: int = 10
    prior_estimation_mode: str = "structure_only"
    use_consistency_feedback: bool = True
    use_preference_restart: bool = True
    inject_preference_guidance: bool = True
    fcpe_actual_variant: str = "default"
    preference_guidance_policy: str = "current_full_search"
    history_prompt_mode: str = "current"
    consult_on_final_generation: bool = False
    learning_query_allocation: str = "paper_11_plus_1"
    standard_crowding_between_interactions: bool = False
    preference_injection_mode: str = "full_tlr"
    fixed_guidance_strength: Optional[float] = None
    guidance_ratio: float = field(default_factory=_default_session_guidance_ratio)
    preference_restart_bias_ratio: float = field(default_factory=_default_session_restart_bias)
    objective_context_mode: str = "basic"
    dm_noise_tau: float = 0.0
    noise_kappa: float = -1.0
    use_legacy_structure_mapping: bool = False
    output_dir: str = "agentic_outputs"
    save_round_images: bool = False
    map_path: Optional[str] = None
    # Prompt defaults: keep a short history summary, but drop noisier evidence blocks.
    use_chosen_vs_rejected: bool = False
    use_history_context: bool = True
    use_behavior_diagnostics: bool = False
    history_max_rounds: int = 2
    # Token budget: thinking/reasoning models consume max_tokens for CoT before outputting
    # Set higher (e.g. 8000) when using reasoning models (gpt-o*, gpt-5*, deepseek-r*)
    llm_max_tokens_structure: int = 8000  # for _generate_structure_only_estimate_llm
    utility_learner_variant: str = "anchored_residual"
    utility_learner_kwargs: Dict[str, Any] = field(default_factory=dict)
    utility_ensemble_mode: str = "off"
    shadow_prior_member_count: int = 5
    temporal_weight_ratio: float = 5.0
    utility_parameter_shadow_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    query_pool_mode: str = "pareto_front"
    query_selector_mode: str = "fcpe_style"
    # Optional pre-computed reference bounds for the utility learner.
    # When provided, the learner's normalisation envelope is set from these
    # fixed bounds instead of the first-round Pareto snapshot, so that
    # training-DM scoring and final evaluation scoring share the same reference.
    # Format: {"archive_ref_mins": [...], "archive_ref_maxs": [...]}
    fixed_reference_bounds: Optional[Dict[str, Any]] = None
    # Symmetric margin for references learned from the first in-loop front.
    # Explicit fixed bounds are not expanded.  Zero preserves legacy behavior.
    reference_bounds_padding_ratio: float = 0.0
    # Widen the in-loop reference to cover the first round's actual query pool.
    # Menus drawn from the critical front can contain dominated solutions that
    # lie outside front zero, which a Pareto-only envelope clips to 0 or 100.
    reference_covers_query_pool: bool = False
    # A value greater than one enables shortlist + final-DM decision output.
    final_shortlist_size: int = 1

    @property
    def total_gen_budget(self) -> int:
        return self.n_gen_warmup + self.n_gen_per_interaction * max(0, self.rounds - 1)

"""Minimal public API and CLI for the simplified preference-agent mainline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Optional, Sequence, Tuple

import numpy as np

from benchmark_defaults import (
    FORMAL_PROPOSED_SESSION_KWARGS,
    resolve_fcpe_actual_variant_runtime_kwargs,
)
from config import DEFAULT_LLM_MODEL

from .models import SessionConfig, SessionResult
from .human_choice import ChoiceProvider
from .session import run_proposed_session
from .utils import ensure_project_ready, normalize_weights
from preference_utility import PAIRWISE_UTILITY_VARIANTS

_DEFAULT_GUIDANCE_RATIO = float(FORMAL_PROPOSED_SESSION_KWARGS["guidance_ratio"])
_DEFAULT_PREF_RESTART_BIAS = float(FORMAL_PROPOSED_SESSION_KWARGS["preference_restart_bias_ratio"])


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    return obj


def _parse_utility_kwargs_json(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --utility-kwargs-json: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--utility-kwargs-json must decode to a JSON object.")
    return dict(parsed)


def run_full_session(
    true_weights: Optional[Sequence[float]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    rounds: int = 4,
    output_dir: str = "agentic_outputs",
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_temperature: float = 0.0,
    query_n_select: int = 5,
    n_gen_warmup: int = 20,
    n_gen_per_interaction: int = 10,
    prior_estimation_mode: str = "structure_only",
    use_consistency_feedback: bool = True,
    use_preference_restart: bool = True,
    inject_preference_guidance: bool = True,
    preference_injection_mode: str = "full_tlr",
    fixed_guidance_strength: float | None = None,
    guidance_ratio: float = _DEFAULT_GUIDANCE_RATIO,
    preference_restart_bias_ratio: float = _DEFAULT_PREF_RESTART_BIAS,
    objective_context_mode: str = "basic",
    dm_noise_tau: float = 0.0,
    noise_kappa: float = -1.0,
    use_legacy_structure_mapping: bool = False,
    use_chosen_vs_rejected: bool = False,
    use_history_context: bool = True,
    use_behavior_diagnostics: bool = False,
    history_max_rounds: int = 2,
    llm_max_tokens_structure: int = 8000,
    utility_learner_variant: str = "anchored_residual",
    utility_learner_kwargs: dict | None = None,
    utility_ensemble_mode: str = "off",
    shadow_prior_member_count: int = 5,
    temporal_weight_ratio: float = 5.0,
    utility_parameter_shadow_configs: dict | None = None,
    query_pool_mode: str = "pareto_front",
    query_selector_mode: str = "fcpe_style",
    fcpe_actual_variant: str = "paper_mainline",
    preference_guidance_policy: str = "current_full_search",
    history_prompt_mode: str = "current",
    consult_on_final_generation: bool = False,
    learning_query_allocation: str = "paper_11_plus_1",
    standard_crowding_between_interactions: bool = False,
    fixed_reference_bounds: dict | None = None,
    reference_bounds_padding_ratio: float = 0.0,
    reference_covers_query_pool: bool = False,
    final_shortlist_size: int = 1,
    choice_provider: Optional[ChoiceProvider] = None,
) -> SessionResult:
    ensure_project_ready()
    if true_weights is None and choice_provider is None:
        raise ValueError(
            "true_weights may be omitted only when a ChoiceProvider supplies "
            "the participant's decisions"
        )
    cfg = SessionConfig(
        rounds=rounds,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        query_n_select=query_n_select,
        n_gen_warmup=n_gen_warmup,
        n_gen_per_interaction=n_gen_per_interaction,
        prior_estimation_mode=prior_estimation_mode,
        use_consistency_feedback=use_consistency_feedback,
        use_preference_restart=use_preference_restart,
        inject_preference_guidance=inject_preference_guidance,
        fcpe_actual_variant=fcpe_actual_variant,
        preference_guidance_policy=preference_guidance_policy,
        history_prompt_mode=history_prompt_mode,
        consult_on_final_generation=consult_on_final_generation,
        learning_query_allocation=learning_query_allocation,
        standard_crowding_between_interactions=standard_crowding_between_interactions,
        preference_injection_mode=preference_injection_mode,
        fixed_guidance_strength=fixed_guidance_strength,
        guidance_ratio=guidance_ratio,
        preference_restart_bias_ratio=preference_restart_bias_ratio,
        objective_context_mode=objective_context_mode,
        dm_noise_tau=dm_noise_tau,
        noise_kappa=noise_kappa,
        use_legacy_structure_mapping=use_legacy_structure_mapping,
        use_chosen_vs_rejected=use_chosen_vs_rejected,
        use_history_context=use_history_context,
        use_behavior_diagnostics=use_behavior_diagnostics,
        history_max_rounds=history_max_rounds,
        llm_max_tokens_structure=llm_max_tokens_structure,
        utility_learner_variant=utility_learner_variant,
        utility_learner_kwargs=dict(utility_learner_kwargs or {}),
        utility_ensemble_mode=utility_ensemble_mode,
        shadow_prior_member_count=int(shadow_prior_member_count),
        temporal_weight_ratio=float(temporal_weight_ratio),
        utility_parameter_shadow_configs=dict(utility_parameter_shadow_configs or {}),
        query_pool_mode=query_pool_mode,
        query_selector_mode=query_selector_mode,
        output_dir=output_dir,
        fixed_reference_bounds=fixed_reference_bounds,
        reference_bounds_padding_ratio=reference_bounds_padding_ratio,
        reference_covers_query_pool=reference_covers_query_pool,
        final_shortlist_size=final_shortlist_size,
    )
    return run_proposed_session(
        cfg,
        tuple(start),
        tuple(end),
        None if true_weights is None else list(normalize_weights(true_weights)),
        interactive=choice_provider is not None,
        choice_provider=choice_provider,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simplified preference-learning session.")
    parser.add_argument("--start", nargs=2, type=int, required=True)
    parser.add_argument("--end", nargs=2, type=int, required=True)
    parser.add_argument("--true-weights", nargs=4, type=float, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--query-n-select", type=int, default=5)
    parser.add_argument("--n-gen-warmup", type=int, default=20)
    parser.add_argument("--n-gen-interact", type=int, default=10)
    parser.add_argument(
        "--prior-estimation-mode",
        choices=["utility_only", "structure_only"],
        default="structure_only",
    )
    parser.add_argument("--no-consistency-feedback", action="store_true")
    parser.add_argument("--no-preference-restart", action="store_true")
    parser.add_argument("--no-preference-guidance", action="store_true")
    parser.add_argument(
        "--fcpe-actual-variant",
        choices=["paper_mainline", "default", "bench_query_slfn", "bench_query_env_restart"],
        default="paper_mainline",
    )
    parser.add_argument(
        "--preference-injection-mode",
        choices=["full_tlr", "env_only", "utility_objective"],
        default="full_tlr",
    )
    parser.add_argument("--fixed-guidance-strength", type=float, default=None)
    parser.add_argument("--guidance-ratio", type=float, default=_DEFAULT_GUIDANCE_RATIO)
    parser.add_argument(
        "--preference-restart-bias-ratio",
        type=float,
        default=_DEFAULT_PREF_RESTART_BIAS,
    )
    parser.add_argument(
        "--objective-context-mode",
        choices=["basic", "formula_card", "formula_card_dmdef"],
        default="basic",
    )
    parser.add_argument("--dm-noise-tau", type=float, default=0.0)
    parser.add_argument(
        "--noise-kappa",
        type=float,
        default=-1.0,
        help="BT-flip DM noise; kappa < 0 disables noise, kappa >= 0 uses exp(-kappa * |delta psi|).",
    )
    parser.add_argument("--legacy-structure-mapping", action="store_true")
    parser.add_argument("--output-dir", type=str, default="agentic_outputs")
    parser.add_argument(
        "--query-pool-mode",
        choices=["pareto_front", "critical_front"],
        default=None,
    )
    parser.add_argument(
        "--query-selector-mode",
        choices=["fcpe_style", "simple", "max_distance"],
        default=None,
    )
    parser.add_argument(
        "--utility-learner-variant",
        choices=list(PAIRWISE_UTILITY_VARIANTS),
        default=None,
    )
    parser.add_argument(
        "--utility-kwargs-json",
        type=str,
        default=None,
        help="JSON object forwarded to the utility learner, e.g. '{\"basis_degree\": 4}'.",
    )
    parser.add_argument(
        "--utility-ensemble-mode",
        choices=["off", "shadow_final", "full_pipeline"],
        default="off",
    )
    parser.add_argument("--json", action="store_true", help="Print the full SessionResult JSON.")
    args = parser.parse_args(argv)
    try:
        args.utility_learner_kwargs = (
            None
            if args.utility_kwargs_json is None
            else _parse_utility_kwargs_json(args.utility_kwargs_json)
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    variant_kwargs = resolve_fcpe_actual_variant_runtime_kwargs(
        args.fcpe_actual_variant,
        query_pool_mode=args.query_pool_mode,
        query_selector_mode=args.query_selector_mode,
        utility_learner_variant=args.utility_learner_variant,
        utility_learner_kwargs=args.utility_learner_kwargs,
    )
    result = run_full_session(
        true_weights=args.true_weights,
        start=tuple(args.start),
        end=tuple(args.end),
        rounds=args.rounds,
        output_dir=args.output_dir,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        query_n_select=args.query_n_select,
        n_gen_warmup=args.n_gen_warmup,
        n_gen_per_interaction=args.n_gen_interact,
        prior_estimation_mode=args.prior_estimation_mode,
        use_consistency_feedback=not args.no_consistency_feedback,
        use_preference_restart=not args.no_preference_restart,
        inject_preference_guidance=not args.no_preference_guidance,
        preference_injection_mode=args.preference_injection_mode,
        fixed_guidance_strength=args.fixed_guidance_strength,
        guidance_ratio=args.guidance_ratio,
        preference_restart_bias_ratio=args.preference_restart_bias_ratio,
        objective_context_mode=args.objective_context_mode,
        dm_noise_tau=args.dm_noise_tau,
        noise_kappa=args.noise_kappa,
        use_legacy_structure_mapping=args.legacy_structure_mapping,
        utility_learner_variant=variant_kwargs.get("utility_learner_variant", "anchored_residual"),
        utility_learner_kwargs=variant_kwargs.get("utility_learner_kwargs"),
        utility_ensemble_mode=args.utility_ensemble_mode,
        query_pool_mode=variant_kwargs.get("query_pool_mode", "pareto_front"),
        query_selector_mode=variant_kwargs.get("query_selector_mode", "fcpe_style"),
        fcpe_actual_variant=str(variant_kwargs.get("fcpe_actual_variant", args.fcpe_actual_variant)),
        preference_guidance_policy=str(
            variant_kwargs.get("preference_guidance_policy", "current_full_search")
        ),
        history_prompt_mode=str(variant_kwargs.get("history_prompt_mode", "current")),
        consult_on_final_generation=bool(
            variant_kwargs.get("consult_on_final_generation", False)
        ),
        standard_crowding_between_interactions=bool(
            variant_kwargs.get("standard_crowding_between_interactions", False)
        ),
    )
    if args.json:
        print(json.dumps(_to_jsonable(asdict(result)), indent=2, ensure_ascii=False))
        return

    print(f"method={result.method}")
    print(f"final_weights={np.round(result.final_weights, 4).tolist()}")
    if result.true_weights is not None:
        print(f"true_weights={np.round(result.true_weights, 4).tolist()}")
        print(
            f"weight_l1={result.weight_l1_error:.4f} "
            f"weight_cos={result.weight_cosine:.4f}"
        )
    print(f"choice_acc={result.choice_reproduction_acc:.4f}")
    print(f"pairwise_loss={result.history_pairwise_loss:.4f}")


if __name__ == "__main__":
    main()

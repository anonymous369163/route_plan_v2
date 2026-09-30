"""Shared benchmark defaults for the current proposed method and fair budgets.

Tunable FCPE preference-injection knobs live only in ``FORMAL_PROPOSED_SESSION_KWARGS``
(``guidance_ratio``, ``preference_restart_bias_ratio``). Other modules read them via imports
or ``build_proposed_run_full_session_kwargs`` / method specs — do not duplicate literals.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

from config import DEFAULT_LLM_MODEL, LLM_BACKEND, OPENAI_BASE_URL

# Kept as alias so existing imports keep working; follows config.LLM_BACKEND
FORMAL_DEFAULT_LLM_MODEL = DEFAULT_LLM_MODEL
PAPER_CONFIG_CONTRACT_VERSION = "tlr_paper_v1_120g_12q_4methods"
FORMAL_MAIN_METHODS = ("Default", "PINSGA2", "IEMOLTR", "Proposed")

FORMAL_PROPOSED_RUNTIME: Dict[str, Any] = {
    "rounds": 12,
    "n_gen_warmup": 10,
    "n_gen_per_interaction": 10,
    "query_n_select": 5,
    "dm_noise_tau": 0.0,
    "noise_kappa": -1.0,
}

# Single source of truth for FCPE / proposed preference-injection knobs.
# guidance_ratio / preference_restart_bias_ratio: joint optimum for mean model_selection_psi
# on the paper-grade guidance×restart sensitivity run (2026-03-31, 24 scenarios x 20 cells);
# best cell g=0.70|r=0.40 (model_selection_psi=3.928, p=0.037 vs prior default g=0.90|r=0.25).
# search_psi optimum differs (g=0.30|r=0.25). See verify_guidance_restart_sensitivity summary.
FORMAL_PROPOSED_SESSION_KWARGS: Dict[str, Any] = {
    "prior_estimation_mode": "structure_only",
    "use_consistency_feedback": False,
    "use_preference_restart": True,
    "inject_preference_guidance": True,
    "fcpe_actual_variant": "paper_mainline",
    "preference_guidance_policy": "benchmark_env_consult_only",
    "history_prompt_mode": "benchmark_runner_up",
    "objective_context_mode": "formula_card_dmdef",
    "guidance_ratio": 1.0,
    "preference_restart_bias_ratio": 0.40,
    "use_legacy_structure_mapping": False,
    "use_chosen_vs_rejected": False,
    "use_history_context": True,
    "use_behavior_diagnostics": False,
    "history_max_rounds": 2,
    "llm_temperature": 0.0,
    "fixed_guidance_strength": None,
    "consult_on_final_generation": True,
    "standard_crowding_between_interactions": True,
    "query_pool_mode": "critical_front",
    "query_selector_mode": "simple",
    "utility_learner_variant": "benchmark_slfn",
    "utility_learner_kwargs": {
        "lambda_delta": 2.0,
        "lambda_cross": 0.05,
        "n_hidden": 16,
        "early_cross_multiplier": 1.0,
    },
}

COMPARE_GUIDANCE_STRENGTH = 0.15
COMPARE_DM_QUERY_SIZE = int(FORMAL_PROPOSED_RUNTIME["query_n_select"])


def get_compare_budget() -> Dict[str, int]:
    rounds = int(FORMAL_PROPOSED_RUNTIME["rounds"])
    n_gen_warmup = int(FORMAL_PROPOSED_RUNTIME["n_gen_warmup"])
    n_gen_per_interaction = int(FORMAL_PROPOSED_RUNTIME["n_gen_per_interaction"])
    # The paper consults after generations 10, 20, ..., 120.  ``rounds`` is
    # therefore the number of consultations, while the first consultation is
    # reached by the warm-up block itself.
    n_interactions = max(0, rounds)
    total_gen = n_gen_warmup + max(0, rounds - 1) * n_gen_per_interaction
    return {
        "rounds": rounds,
        "n_gen_warmup": n_gen_warmup,
        "n_gen_per_interaction": n_gen_per_interaction,
        "n_interactions": n_interactions,
        "total_gen": total_gen,
        "query_n_select": int(FORMAL_PROPOSED_RUNTIME["query_n_select"]),
    }


def build_formal_structure_method_spec(name: str = "SO") -> Dict[str, Any]:
    return {
        "name": name,
        "prior_estimation_mode": FORMAL_PROPOSED_SESSION_KWARGS["prior_estimation_mode"],
        "use_consistency_feedback": FORMAL_PROPOSED_SESSION_KWARGS["use_consistency_feedback"],
        "use_preference_restart": FORMAL_PROPOSED_SESSION_KWARGS["use_preference_restart"],
        "inject_preference_guidance": FORMAL_PROPOSED_SESSION_KWARGS["inject_preference_guidance"],
        "objective_context_mode": FORMAL_PROPOSED_SESSION_KWARGS["objective_context_mode"],
        "guidance_ratio": FORMAL_PROPOSED_SESSION_KWARGS["guidance_ratio"],
        "preference_restart_bias_ratio": FORMAL_PROPOSED_SESSION_KWARGS["preference_restart_bias_ratio"],
        "use_legacy_structure_mapping": FORMAL_PROPOSED_SESSION_KWARGS["use_legacy_structure_mapping"],
        "use_chosen_vs_rejected": FORMAL_PROPOSED_SESSION_KWARGS["use_chosen_vs_rejected"],
        "use_history_context": FORMAL_PROPOSED_SESSION_KWARGS["use_history_context"],
        "use_behavior_diagnostics": FORMAL_PROPOSED_SESSION_KWARGS["use_behavior_diagnostics"],
        "history_max_rounds": FORMAL_PROPOSED_SESSION_KWARGS["history_max_rounds"],
        "llm_temperature": FORMAL_PROPOSED_SESSION_KWARGS["llm_temperature"],
        "fixed_guidance_strength": FORMAL_PROPOSED_SESSION_KWARGS["fixed_guidance_strength"],
        "query_n_select": FORMAL_PROPOSED_RUNTIME["query_n_select"],
    }


def build_formal_structure_utility_only_method_spec(name: str = "UO") -> Dict[str, Any]:
    return {
        "name": name,
        "prior_estimation_mode": "utility_only",
        "use_consistency_feedback": False,
        "use_preference_restart": True,
        "inject_preference_guidance": True,
        "objective_context_mode": "basic",
        "query_n_select": int(FORMAL_PROPOSED_RUNTIME["query_n_select"]),
        "guidance_ratio": float(FORMAL_PROPOSED_SESSION_KWARGS["guidance_ratio"]),
        "preference_restart_bias_ratio": float(
            FORMAL_PROPOSED_SESSION_KWARGS["preference_restart_bias_ratio"]
        ),
        "use_chosen_vs_rejected": False,
        "use_history_context": True,
        "use_behavior_diagnostics": False,
        "history_max_rounds": 2,
    }


def build_formal_structure_method_specs() -> list[Dict[str, Any]]:
    return [
        build_formal_structure_utility_only_method_spec(),
        build_formal_structure_method_spec(),
    ]


def build_formal_structure_actual_variant_method_specs(
    fcpe_actual_variant: str = "bench_query_env_restart",
) -> list[Dict[str, Any]]:
    variant = str(fcpe_actual_variant or "bench_query_env_restart")
    specs = [
        dict(build_formal_structure_utility_only_method_spec(), analysis_role="base"),
        dict(build_formal_structure_method_spec(), analysis_role="challenger"),
    ]
    if variant == "default":
        return specs

    resolved: list[Dict[str, Any]] = []
    for spec in specs:
        updated = dict(spec)
        updated.update(get_fcpe_actual_variant_defaults(variant))
        resolved.append(updated)
    return resolved




def get_formal_structure_benchmark_defaults() -> Dict[str, Any]:
    return {
        "llm_model": FORMAL_DEFAULT_LLM_MODEL,
        "rounds": int(FORMAL_PROPOSED_RUNTIME["rounds"]),
        "n_gen_warmup": int(FORMAL_PROPOSED_RUNTIME["n_gen_warmup"]),
        "n_gen_interact": int(FORMAL_PROPOSED_RUNTIME["n_gen_per_interaction"]),
        "noise_kappas": [-1.0, 1.0],
        "weight_presets": ["wr_a10"],
        "cases_per_weight_preset": 30,
        "seed": 42,
        "fcpe_actual_variant": "bench_query_env_restart",
        "methods": build_formal_structure_actual_variant_method_specs(
            "bench_query_env_restart"
        ),
    }


def build_fcpe_full_method_spec(name: str = "Full") -> Dict[str, Any]:
    return {
        "name": name,
        "prior_estimation_mode": "structure_only",
        "use_consistency_feedback": True,
        "use_preference_restart": True,
        "inject_preference_guidance": True,
        "guidance_ratio": float(FORMAL_PROPOSED_SESSION_KWARGS["guidance_ratio"]),
        "preference_restart_bias_ratio": float(
            FORMAL_PROPOSED_SESSION_KWARGS["preference_restart_bias_ratio"]
        ),
    }


def build_fcpe_full_env_only_method_spec(name: str = "Full-EnvOnly") -> Dict[str, Any]:
    spec = build_fcpe_full_method_spec(name=name)
    spec.update(
        {
            "use_preference_restart": False,
            "preference_injection_mode": "env_only",
            "guidance_ratio": 1.0,
        }
    )
    return spec


def build_fcpe_mainline_method_specs() -> list[Dict[str, Any]]:
    return [
        {
            "name": "UtilityOnly",
            "prior_estimation_mode": "utility_only",
            "use_consistency_feedback": False,
            "use_preference_restart": True,
            "inject_preference_guidance": True,
            "use_chosen_vs_rejected": False,
            "use_history_context": True,
            "use_behavior_diagnostics": False,
            "history_max_rounds": 2,
            "guidance_ratio": float(FORMAL_PROPOSED_SESSION_KWARGS["guidance_ratio"]),
            "preference_restart_bias_ratio": float(
                FORMAL_PROPOSED_SESSION_KWARGS["preference_restart_bias_ratio"]
            ),
        },
        {
            "name": "UtilityOnly+LLM-Structure",
            "prior_estimation_mode": "structure_only",
            "use_consistency_feedback": True,
            "use_preference_restart": True,
            "inject_preference_guidance": True,
            "use_chosen_vs_rejected": False,
            "use_history_context": True,
            "use_behavior_diagnostics": False,
            "history_max_rounds": 2,
            "guidance_ratio": float(FORMAL_PROPOSED_SESSION_KWARGS["guidance_ratio"]),
            "preference_restart_bias_ratio": float(
                FORMAL_PROPOSED_SESSION_KWARGS["preference_restart_bias_ratio"]
            ),
        },
    ]


def build_fcpe_guidance_ablation_method_specs() -> list[Dict[str, Any]]:
    return [
        build_fcpe_full_method_spec(),
        build_fcpe_full_env_only_method_spec(),
    ]


def get_fcpe_actual_variant_defaults(fcpe_actual_variant: str) -> Dict[str, Any]:
    variant = str(fcpe_actual_variant or "default")
    if variant == "default":
        return {
            "fcpe_actual_variant": "default",
            "query_pool_mode": "pareto_front",
            "query_selector_mode": "fcpe_style",
            "utility_learner_variant": "anchored_residual",
            "utility_learner_kwargs": {},
            "preference_guidance_policy": "current_full_search",
            "history_prompt_mode": "current",
        }

    base: Dict[str, Any] = {
        "fcpe_actual_variant": variant,
        "query_pool_mode": "critical_front",
        "query_selector_mode": "simple",
        "query_n_select": int(FORMAL_PROPOSED_RUNTIME["query_n_select"]),
        "utility_learner_variant": "benchmark_slfn",
        "utility_learner_kwargs": {},
    }
    if variant == "paper_mainline":
        base.update(
            {
                "utility_learner_kwargs": {
                    "lambda_delta": 2.0,
                    "lambda_cross": 0.05,
                    "n_hidden": 16,
                    "early_cross_multiplier": 1.0,
                },
                "preference_guidance_policy": "benchmark_env_consult_only",
                "history_prompt_mode": "benchmark_runner_up",
                "use_preference_restart": True,
                "guidance_ratio": 1.0,
                "use_consistency_feedback": False,
                "consult_on_final_generation": True,
                "standard_crowding_between_interactions": True,
            }
        )
        return base
    if variant == "bench_query_slfn":
        base["use_preference_restart"] = True
        return base
    if variant == "bench_query_env_restart":
        base.update(
            {
                "preference_guidance_policy": "benchmark_env_consult_only",
                "history_prompt_mode": "benchmark_runner_up",
                "use_preference_restart": True,
                "guidance_ratio": 1.0,
                "use_consistency_feedback": False,
            }
        )
        return base
    raise ValueError(f"Unknown FCPE actual variant: {variant}")


def resolve_fcpe_actual_variant_runtime_kwargs(
    fcpe_actual_variant: str,
    **explicit_overrides: Any,
) -> Dict[str, Any]:
    resolved = get_fcpe_actual_variant_defaults(fcpe_actual_variant)
    for key, value in explicit_overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def _apply_fcpe_bench_query_slfn_overrides(
    spec: Dict[str, Any],
    *,
    name: str,
) -> Dict[str, Any]:
    updated = dict(spec)
    updated.update(get_fcpe_actual_variant_defaults("bench_query_slfn"))
    updated["name"] = name
    return updated


def build_fcpe_bench_query_slfn_mainline_method_specs() -> list[Dict[str, Any]]:
    return [
        _apply_fcpe_bench_query_slfn_overrides(
            build_formal_structure_utility_only_method_spec(),
            name="UO-BenchQuery",
        ),
        _apply_fcpe_bench_query_slfn_overrides(
            build_formal_structure_method_spec(),
            name="SO-BenchQuery",
        ),
    ]


def build_fcpe_bench_query_slfn_guidance_ablation_method_specs() -> list[Dict[str, Any]]:
    return [
        _apply_fcpe_bench_query_slfn_overrides(
            build_fcpe_full_method_spec(),
            name="Full-BenchQuery",
        ),
        _apply_fcpe_bench_query_slfn_overrides(
            build_fcpe_full_env_only_method_spec(),
            name="Full-EnvOnly-BenchQuery",
        ),
    ]


def _apply_fcpe_bench_query_env_restart_overrides(
    spec: Dict[str, Any],
    *,
    name: str,
) -> Dict[str, Any]:
    updated = dict(spec)
    updated.update(get_fcpe_actual_variant_defaults("bench_query_env_restart"))
    updated["name"] = name
    return updated


def build_fcpe_bench_query_env_restart_mainline_method_specs() -> list[Dict[str, Any]]:
    return [
        _apply_fcpe_bench_query_env_restart_overrides(
            build_formal_structure_utility_only_method_spec(),
            name="UO-BenchEnv",
        ),
        _apply_fcpe_bench_query_env_restart_overrides(
            build_formal_structure_method_spec(),
            name="SO-BenchEnv",
        ),
    ]


def build_fcpe_bench_query_env_restart_guidance_ablation_method_specs() -> list[Dict[str, Any]]:
    return [
        _apply_fcpe_bench_query_env_restart_overrides(
            build_fcpe_full_method_spec(),
            name="Full-BenchEnv",
        ),
        _apply_fcpe_bench_query_env_restart_overrides(
            build_fcpe_full_env_only_method_spec(),
            name="Full-EnvOnly-BenchEnv",
        ),
    ]


def validate_run_full_session_kwargs(kwargs: Dict[str, Any]) -> None:
    from preference_agent import run_full_session

    supported = set(inspect.signature(run_full_session).parameters.keys())
    unsupported = sorted(set(kwargs.keys()) - supported)
    if unsupported:
        raise ValueError(
            "Unsupported run_full_session kwargs in benchmark defaults: "
            + ", ".join(unsupported)
        )


def build_proposed_run_full_session_kwargs(
    *,
    llm_model: str,
    output_dir: str,
) -> Dict[str, Any]:
    kwargs = {
        "llm_model": llm_model,
        "output_dir": output_dir,
        **FORMAL_PROPOSED_RUNTIME,
        **FORMAL_PROPOSED_SESSION_KWARGS,
    }
    validate_run_full_session_kwargs(kwargs)
    return kwargs


def benchmark_config_snapshot(llm_model: str = FORMAL_DEFAULT_LLM_MODEL) -> Dict[str, Any]:
    return {
        "paper_config_contract_version": PAPER_CONFIG_CONTRACT_VERSION,
        "formal_main_methods": list(FORMAL_MAIN_METHODS),
        "llm_backend": LLM_BACKEND,
        "openai_base_url": OPENAI_BASE_URL,
        "llm_model": llm_model,
        "formal_proposed_runtime": dict(FORMAL_PROPOSED_RUNTIME),
        "formal_proposed_session_kwargs": dict(FORMAL_PROPOSED_SESSION_KWARGS),
        "formal_structure_method_spec": build_formal_structure_method_spec(),
        "compare_budget": get_compare_budget(),
        "compare_guidance_strength": float(COMPARE_GUIDANCE_STRENGTH),
        "compare_dm_query_size": int(COMPARE_DM_QUERY_SIZE),
    }

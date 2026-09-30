"""LLM structure-prior helpers for the simplified preference-agent mainline."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import DIRECTIONS, OBJECTIVES
from .llm_interface import maybe_get_client
from .llm_trace import current_llm_trace
from .models import RoundTrace, SessionConfig
from .scoring import (
    _find_choice_idx,
    _format_choice_comparison_prompt,
    _format_relative_metrics_list,
)
try:
    # Prompt-language experiments only; the default English prompt is sent unchanged.
    from .prompt_adaptation import adapt_messages
except ImportError:
    def adapt_messages(messages):
        return messages
from .utils import (
    extract_json_from_text,
    format_metrics_list,
    objective_context_card_text,
    objective_descriptions_text,
    objective_reasoning_guidance_text,
)


def _runner_up_summary(trace: RoundTrace) -> tuple[Optional[str], Optional[str]]:
    menu = np.asarray(trace.presented_metrics, dtype=float)
    if menu.ndim == 1:
        menu = menu[None, :]
    if len(menu) <= 1:
        return None, None

    chosen_idx = int(np.clip(trace.choice_idx, 0, len(menu) - 1))
    mins = menu.min(axis=0)
    rng = menu.max(axis=0) - mins + 1e-12
    rel = (menu - mins) / rng
    max_mask = np.array([direction == "max" for direction in DIRECTIONS], dtype=bool)
    if np.any(max_mask):
        rel[:, max_mask] = 1.0 - rel[:, max_mask]
    chosen_rel = rel[chosen_idx]

    best_dist = np.inf
    runner_idx = -1
    for idx in range(len(menu)):
        if idx == chosen_idx:
            continue
        dist = float(np.sum((rel[idx] - chosen_rel) ** 2))
        if dist < best_dist:
            best_dist = dist
            runner_idx = idx
    if runner_idx < 0:
        return None, None

    runner = menu[runner_idx]
    runner_vals = ", ".join(f"{OBJECTIVES[k]}={runner[k]:.4f}" for k in range(len(OBJECTIVES)))
    tradeoffs: List[str] = []
    chosen = menu[chosen_idx]
    for idx, name in enumerate(OBJECTIVES):
        higher_is_better = DIRECTIONS[idx] == "max"
        if higher_is_better:
            if chosen[idx] > runner[idx] + 1e-6:
                tradeoffs.append(f"better {name}")
            elif chosen[idx] < runner[idx] - 1e-6:
                tradeoffs.append(f"worse {name}")
        else:
            if chosen[idx] < runner[idx] - 1e-6:
                tradeoffs.append(f"better {name}")
            elif chosen[idx] > runner[idx] + 1e-6:
                tradeoffs.append(f"worse {name}")
    return runner_vals, (", ".join(tradeoffs) if tradeoffs else "similar")


def _format_history_context(
    traces: Sequence[RoundTrace],
    max_recent_rounds: int = 2,
    mode: str = "current",
) -> str:
    if not traces:
        return ""

    max_recent_rounds = max(1, int(max_recent_rounds))
    recent = list(traces[-max_recent_rounds:])
    current_weights = np.round(np.asarray(traces[-1].method_weights_after, dtype=float), 3).tolist()
    current_focus = "+".join(traces[-1].focus) if traces[-1].focus else "?"

    lines = [
        "Interaction summary:",
        "  Current provisional model estimate (context only; not user evidence): "
        f"weights={current_weights}; focus=[{current_focus}]",
        "  Recent rounds:",
    ]
    start_idx = len(traces) - len(recent) + 1
    for idx, trace in enumerate(recent, start=start_idx):
        after = np.round(np.asarray(trace.method_weights_after, dtype=float), 3).tolist()
        focus_text = "+".join(trace.focus) if trace.focus else "?"
        if str(mode) == "benchmark_runner_up":
            chosen = np.asarray(trace.presented_metrics, dtype=float)
            chosen = chosen[trace.choice_idx] if len(chosen) > int(trace.choice_idx) else np.asarray([])
            chosen_vals = (
                ", ".join(f"{OBJECTIVES[k]}={chosen[k]:.4f}" for k in range(len(OBJECTIVES)))
                if chosen.size == len(OBJECTIVES)
                else f"path {trace.choice_idx + 1}"
            )
            lines.append(
                f"    Round {idx}: chose [{chosen_vals}]; focus=[{focus_text}]; updated estimate={after}"
            )
            runner_vals, tradeoff = _runner_up_summary(trace)
            if runner_vals is not None:
                lines.append(f"              over closest alternative [{runner_vals}]")
                lines.append(f"              (trade-off: {tradeoff})")
        else:
            lines.append(
                f"    Round {idx}: chose path {trace.choice_idx + 1}; "
                f"focus=[{focus_text}]; updated estimate={after}"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_importance_order(raw_items: Sequence[Any]) -> Optional[List[Tuple[str, str]]]:
    """Parse importance ordering pairs: each [higher, lower] means w_higher > w_lower."""
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return None
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for item in raw_items:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            return None
        higher = str(item[0]).strip()
        lower = str(item[1]).strip()
        if higher not in OBJECTIVES or lower not in OBJECTIVES or higher == lower:
            return None
        key = (higher, lower)
        if key not in seen and (lower, higher) not in seen:
            pairs.append(key)
            seen.add(key)
    return pairs


def _parse_interaction_hints(raw_items: Sequence[Any]) -> Optional[List[Dict[str, Any]]]:
    hints: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            return None
        if item.get("type") != "joint_bottleneck":
            return None
        pair = item.get("pair")
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            return None
        left = str(pair[0]).strip()
        right = str(pair[1]).strip()
        if left not in OBJECTIVES or right not in OBJECTIVES or left == right:
            return None
        try:
            strength = float(item.get("strength"))
        except Exception:
            return None
        if not (0.0 <= strength <= 1.0):
            return None
        hints.append(
            {
                "pair": [left, right],
                "type": "joint_bottleneck",
                "strength": strength,
            }
        )
    return hints


def _generate_structure_only_estimate_llm(
    all_metrics: Sequence[Sequence[float]],
    selected_metrics: Sequence[float],
    cfg: SessionConfig,
    traces: Sequence[RoundTrace] = (),
    consistency_feedback: str = "",
) -> Optional[Dict[str, Any]]:
    selected_idx = _find_choice_idx(all_metrics, selected_metrics)
    history_block = (
        _format_history_context(
            traces,
            max_recent_rounds=getattr(cfg, "history_max_rounds", 2),
            mode=getattr(cfg, "history_prompt_mode", "current"),
        )
        if getattr(cfg, "use_history_context", True)
        else ""
    )
    feedback_block = ""
    if getattr(cfg, "use_behavior_diagnostics", True) and consistency_feedback:
        feedback_block = (
            "\n--- DATA CONSISTENCY FEEDBACK ---\n"
            + consistency_feedback
            + "--- END FEEDBACK ---\n\n"
            "Use the feedback above to correct earlier structural mistakes.\n\n"
        )
    cvr_block = (
        "Chosen-vs-rejected comparisons:\n"
        + _format_choice_comparison_prompt(all_metrics, selected_idx)
        + "\n\n"
    ) if getattr(cfg, "use_chosen_vs_rejected", True) else ""
    prompt = (
        "You are estimating raw sparse preference structure for multi-objective path planning.\n"
        f"Objectives: {objective_descriptions_text()}\n"
        + objective_context_card_text(cfg.objective_context_mode)
        + objective_reasoning_guidance_text()
        + "\n"
        + history_block
        + feedback_block
        + "Candidate paths (raw values):\n"
        + format_metrics_list(all_metrics)
        + "Relative standing in this round (0=best, 1=worst; lower is better after direction alignment):\n"
        + _format_relative_metrics_list(all_metrics)
        + f"User chose path {selected_idx + 1}.\n\n"
        + cvr_block
        + "Treat observed user choices as the only preference evidence. "
        "Previous model estimates are provisional context, not additional evidence.\n"
        + "Return JSON with exactly these keys:\n"
        "{\n"
        '  "importance_order": [["<more_important_objective>", "<less_important_objective>"]],\n'
        '  "interaction_hints": [{"pair": ["<objective>", "<objective>"], "type": "joint_bottleneck", "strength": 0.0-1.0}],\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reasoning": "Brief evidence-based explanation"\n'
        "}\n"
        "importance_order: list of [higher, lower] pairs where the user values 'higher' MORE than 'lower'. "
        "Only include pairs with clear evidence from the choices. Maximum 3 importance pairs.\n"
        "interaction_hints captures pairs for which the observed choices support a joint bottleneck; "
        "candidate co-variation alone is insufficient. Maximum 2 interaction hints.\n"
        "If choice evidence is insufficient, return an empty list for that field rather than guessing.\n"
    )

    import sys
    messages = [
        {
            "role": "system",
            "content": "Extract raw sparse preference structure only and return strict JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    messages = adapt_messages(messages)
    request_record = {
        "call_kind": "structure_only",
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": cfg.llm_temperature,
        "max_tokens": getattr(cfg, "llm_max_tokens_structure", 8000),
    }
    trace = current_llm_trace()
    cached_raw = trace.lookup(request_record) if trace is not None else None
    fresh_response = cached_raw is None
    raw = "" if cached_raw is None else str(cached_raw)

    def finish(value, status: str, warning: str = ""):
        if trace is not None and fresh_response:
            trace.record(request_record, raw, status)
        if warning:
            print(warning, file=sys.stderr, flush=True)
        return value

    if cached_raw is None:
        client = maybe_get_client()
        if client is None:
            return finish(
                None,
                "no_client",
                "[LLM WARN] structure_only: no client available; skipping LLM.",
            )
        try:
            response = client.chat.completions.create(
                model=cfg.llm_model,
                messages=messages,
                temperature=cfg.llm_temperature,
                max_tokens=getattr(cfg, "llm_max_tokens_structure", 8000),
            )
            raw = response.choices[0].message.content or ""
        except Exception as _exc:
            return finish(
                None,
                f"api_error:{type(_exc).__name__}",
                f"[LLM WARN] structure_only: API error ({type(_exc).__name__}: {str(_exc)[:120]}).",
            )

    try:
        parsed = extract_json_from_text(raw)
        required = {"interaction_hints", "confidence", "reasoning"}
        if not isinstance(parsed, dict) or not required.issubset(parsed.keys()):
            return finish(
                None,
                "invalid_missing_fields",
                f"[LLM WARN] structure_only: missing required fields in response (got keys={list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}).",
            )
        interaction_raw = parsed.get("interaction_hints")
        if not isinstance(interaction_raw, Sequence) or isinstance(interaction_raw, (str, bytes)):
            return finish(
                None,
                "invalid_interaction_type",
                f"[LLM WARN] structure_only: interaction_hints is not a list (type={type(interaction_raw).__name__}).",
            )
        interaction_hints = _parse_interaction_hints(interaction_raw)
        if interaction_hints is None or len(interaction_hints) > 2:
            return finish(
                None,
                "invalid_interactions",
                f"[LLM WARN] structure_only: interaction_hints parse failed or too many hints (n={len(interaction_raw)}).",
            )
        importance_raw = parsed.get("importance_order", [])
        importance_order = _parse_importance_order(importance_raw)
        if importance_order is None or len(importance_order) > 3:
            importance_order = []
        confidence = float(parsed.get("confidence"))
        if not (0.0 <= confidence <= 1.0):
            return finish(
                None,
                "invalid_confidence",
                f"[LLM WARN] structure_only: confidence out of range ({confidence}).",
            )
        return finish(
            {
                "interaction_hints": interaction_hints,
                "importance_order": [list(p) for p in importance_order],
                "confidence": confidence,
                "reasoning": str(parsed.get("reasoning", "")),
            },
            "valid",
        )
    except Exception as _exc:
        return finish(
            None,
            f"parse_error:{type(_exc).__name__}",
            f"[LLM WARN] structure_only: parse error ({type(_exc).__name__}: {str(_exc)[:120]}).",
        )

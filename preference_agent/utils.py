"""
General-purpose utilities: math helpers, path/encoding tools, and basic prompt
formatting functions that do NOT depend on metrics normalization.

Functions that need normalize_metrics live in scoring.py to avoid circular imports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .constants import OBJECTIVE_DESCRIPTIONS, OBJECTIVES

try:
    from dsop import decode_solution_to_path, MAX_TOWERS, run_optimizer as _run_optimizer

    TRAINING_IMPORT_ERROR = None
    _TRAINING_AVAILABLE = True
except Exception as e:
    decode_solution_to_path = None
    MAX_TOWERS = 20
    _run_optimizer = None
    TRAINING_IMPORT_ERROR = e
    _TRAINING_AVAILABLE = False


def ensure_project_ready() -> None:
    if TRAINING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import dsop (optimizer / path decode). Run this script in your project environment.\n"
            f"Original error: {TRAINING_IMPORT_ERROR}"
        )


def _decode_candidate_path(
    decision_vars: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]
) -> List[Tuple[int, int]]:
    row = np.asarray(decision_vars, dtype=float).reshape(-1)
    legacy_dim = 1 + 2 * int(MAX_TOWERS)
    if row.size != legacy_dim:
        raise ValueError(
            f"Unsupported decision encoding with {row.size} variables; "
            f"expected legacy path encoding with {legacy_dim} variables."
        )
    return decode_solution_to_path(row, start, end)


def _get_query_candidate_source(
    result: dict,
    *,
    query_pool_mode: str = "pareto_front",
    min_pool_size: int = 1,
) -> Tuple[Sequence[Any], Sequence[Sequence[float]], str, Optional[int]]:
    def _backfill(
        base_x: Sequence[Any],
        base_f: Sequence[Sequence[float]],
        *,
        source: str,
    ) -> Tuple[List[Any], List[List[float]], str]:
        """Deterministically preserve the source pool and fill its shortfall.

        The registered menu size is part of the feedback budget.  A small
        splitting front must therefore not silently turn a comparison round
        into a singleton query.  Pareto candidates are considered first,
        followed by the remaining feasible population.  No RNG or search
        state is touched.
        """
        target = max(1, int(min_pool_size))
        out_x = list(base_x)
        out_f = np.asarray(base_f, dtype=float).reshape(len(out_x), -1).tolist()
        if len(out_x) >= target:
            return out_x, out_f, source

        signatures = {
            np.asarray(row, dtype=float).reshape(-1).tobytes() for row in out_x
        }
        for x_key, f_key in (("pareto_X", "pareto_F"), ("feasible_X", "feasible_F")):
            extra_x = result.get(x_key)
            extra_f = result.get(f_key)
            if extra_x is None or extra_f is None:
                continue
            for row_x, row_f in zip(extra_x, extra_f):
                signature = np.asarray(row_x, dtype=float).reshape(-1).tobytes()
                if signature in signatures:
                    continue
                out_x.append(row_x)
                out_f.append(np.asarray(row_f, dtype=float).reshape(-1).tolist())
                signatures.add(signature)
                if len(out_x) >= target:
                    return out_x, out_f, f"{source}_backfilled"
        return out_x, out_f, (f"{source}_backfilled" if len(out_x) > len(base_x) else source)

    mode = str(query_pool_mode or "pareto_front").strip().lower()
    if mode == "critical_front":
        query_X = result.get("query_X")
        query_F = result.get("query_F")
        if query_X is not None and query_F is not None and len(query_X) > 0 and len(query_F) > 0:
            filled_x, filled_f, source = _backfill(
                query_X,
                query_F,
                source=str(result.get("query_pool_source") or "critical_front"),
            )
            return (
                filled_x,
                filled_f,
                source,
                None if result.get("query_pool_generation") is None else int(result["query_pool_generation"]),
            )
        filled_x, filled_f, source = _backfill(
            result["pareto_X"],
            result["pareto_F"],
            source="pareto_front_fallback",
        )
        return (
            filled_x,
            filled_f,
            source,
            None if result.get("query_pool_generation") is None else int(result["query_pool_generation"]),
        )
    filled_x, filled_f, source = _backfill(
        result["pareto_X"],
        result["pareto_F"],
        source="pareto_front",
    )
    return (
        filled_x,
        filled_f,
        source,
        None if result.get("query_pool_generation") is None else int(result["query_pool_generation"]),
    )


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def normalize_weights(weights: Sequence[float], min_weight: float = 0.01) -> List[float]:
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    if w.size < len(OBJECTIVES):
        w = np.pad(w, (0, len(OBJECTIVES) - w.size), constant_values=0.0)
    elif w.size > len(OBJECTIVES):
        w = w[: len(OBJECTIVES)]
    if float(w.sum()) <= 1e-12:
        w = np.ones(len(w), dtype=float)
    w = w / float(w.sum())
    w = np.maximum(w, min_weight)
    w = w / float(w.sum())
    return w.tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return float(np.dot(a_arr, b_arr) / denom) if denom > 1e-12 else 0.0


def l1_error(a: Sequence[float], b: Sequence[float]) -> float:
    return float(
        np.sum(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    )


def maybe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_json_from_text(text: str) -> Any:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    elif "[" in raw and "]" in raw:
        raw = raw[raw.find("[") : raw.rfind("]") + 1]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Prompt-formatting helpers (no dependency on scoring.normalize_metrics)
# ---------------------------------------------------------------------------

def objective_descriptions_text() -> str:
    return ", ".join(OBJECTIVE_DESCRIPTIONS) + "."


def objective_context_card_text(mode: str = "basic") -> str:
    if mode == "basic":
        return ""
    if mode not in {"formula_card", "formula_card_dmdef"}:
        raise ValueError(
            "objective_context_mode must be one of {'basic', 'formula_card', 'formula_card_dmdef'}"
        )

    lines = [
        "Objective formula card (power transmission line routing):",
        "- User-facing space: [Cost, Length, Safety, Smoothness].",
        "",
        "Cost (lower=better):",
        "  cost = tower_siting_cost(terrain) + line_landuse_premium + crossing_penalty",
        "  Covers construction cost, land-use fees, and obstacle crossing fees.",
        "",
        "Length (lower=better):",
        "  length = sum of Euclidean segment lengths along the path",
        "",
        "Safety (higher=better):",
        "  safety = w_c * clamp(percentile_10(hazard_clearance) / ref, 0, 1)",
        "          + w_e * (1 - mean(hazard_clearance < ref))",
        "  hazard_clearance = distance to nearest safety hazard (routing_coarse: building, road, water).",
        "  Combines 10th-percentile clearance quality and low exposure ratio along the line.",
        "  Safety is a SUBSTANTIVE engineering metric: clearance from built-up, traffic, and water matters",
        "  for EMF, fire/flood risk, and regulatory compliance.",
        "",
        "Smoothness (higher=better):",
        "  smoothness = -bend_penalty",
        "  bend_penalty = sum(0.45 * mild_turn^2 + 1.75 * sharp_turn^2) over all vertices",
        "  Smoothness is a STRUCTURAL constraint: sharp turns require expensive angle towers",
        "  and increase mechanical stress on conductors. A high-smoothness path can cost",
        "  significantly less in tower engineering even if its route length is longer.",
        "",
        "Canonicalization: all four objectives are normalized to [0, 1] before comparison.",
        "After normalization, no objective has inherently larger or smaller scale.",
        "The decision maker can assign high weight to ANY of the four objectives.",
    ]
    return "\n".join(lines) + "\n"


def objective_reasoning_guidance_text() -> str:
    return (
        "Reasoning notes:\n"
        "- Compare objectives using relative standing within the current round; "
        "raw magnitudes across objectives are not directly comparable.\n"
        "- Safety and Smoothness are higher-is-better objectives.\n"
        "- Smoothness means fewer sharp turns, lower cumulative bending, and gentler direction changes.\n"
        "- Smoothness is not the same as Length: a route can be longer but smoother, or shorter but less smooth.\n"
        "- Smoothness values can be numerically small or negative; do not treat Smoothness as unimportant "
        "because its raw scale is smaller than Cost or Length.\n"
    )


def format_metrics(metrics: Sequence[float], precision: int = 2) -> str:
    vals = np.asarray(metrics, dtype=float)
    return ", ".join(
        f"{name}={vals[idx]:.{precision}f}" for idx, name in enumerate(OBJECTIVES)
    )


def format_metrics_list(metrics_list: Sequence[Sequence[float]], precision: int = 2) -> str:
    return "".join(
        f"  {i}. {format_metrics(m, precision=precision)}\n"
        for i, m in enumerate(metrics_list, 1)
    )

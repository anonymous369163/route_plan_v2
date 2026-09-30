"""Objective definitions and lookup constants."""

from __future__ import annotations

from typing import Dict, List

OBJECTIVES = ("Cost", "Length", "Safety", "Smoothness")
FOCUS_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(OBJECTIVES)}

DIRECTIONS = ("min", "min", "max", "max")
DEFAULT_WEIGHTS: List[float] = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

OBJECTIVE_DESCRIPTIONS = (
    "Cost (lower=better; total project/construction cost)",
    "Length (lower=better; total route length)",
    "Safety (higher=better; clearance / low exposure vs buildings, roads, and water on routing_coarse maps)",
    "Smoothness (higher=better; fewer sharp turns, lower cumulative bending, smoother alignment)",
)

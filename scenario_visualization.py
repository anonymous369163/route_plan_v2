#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified visualization API for route planning: terrain map + start/end + one or more paths.

Designed for calls from ``trainingDSOP.run_optimizer`` results or standalone scripts.
Figure text is in English (project convention).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import build_terrain_cmap_for_map

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _normalize_path(path: Optional[Sequence[Sequence[float]]]) -> List[Tuple[int, int]]:
    if path is None:
        return []
    out: List[Tuple[int, int]] = []
    for p in path:
        if p is None:
            continue
        out.append((int(round(float(p[0]))), int(round(float(p[1])))))
    return out


def draw_paths_on_map_axes(
    ax: plt.Axes,
    grid_map: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    path_specs: Sequence[Dict[str, Any]],
    *,
    map_mode: str = "routing_coarse",
) -> None:
    """Draw terrain and paths on an existing Axes.

    Each spec: ``path`` (sequence of points), ``color``, ``label``, optional ``linewidth``, ``alpha``.
    """
    cmap = build_terrain_cmap_for_map(grid_map, map_mode=map_mode)
    ax.imshow(
        grid_map,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=255,
    )
    ax.scatter(
        start[0],
        start[1],
        s=120,
        color="#f59e0b",
        marker="o",
        edgecolor="white",
        linewidth=1.5,
        label="Start",
        zorder=10,
    )
    ax.scatter(
        end[0],
        end[1],
        s=180,
        color="#111827",
        marker="*",
        edgecolor="white",
        linewidth=1.0,
        label="End",
        zorder=10,
    )
    for spec in path_specs:
        pts = _normalize_path(spec.get("path"))
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        ax.plot(
            xs,
            ys,
            color=spec.get("color", "#dc2626"),
            linewidth=float(spec.get("linewidth", 2.4)),
            linestyle=spec.get("linestyle", "-"),
            alpha=float(spec.get("alpha", 0.95)),
            label=spec.get("label", "Path"),
        )
    ax.set_xticks([])
    ax.set_yticks([])


def _default_map_mode() -> str:
    try:
        import dsop as t

        return str(getattr(t, "MAP_MODE", "routing_coarse"))
    except Exception:
        return "routing_coarse"


def load_routing_grid_map() -> np.ndarray:
    """Load the same cached routing map used by the DSOP optimiser."""
    import dsop as t

    return np.asarray(t._get_cached_map(), dtype=np.int32)


def visualize_planning_solution(
    *,
    start: Tuple[int, int],
    end: Tuple[int, int],
    best_path: Sequence[Sequence[float]],
    grid_map: Optional[np.ndarray] = None,
    initial_path: Optional[Sequence[Sequence[float]]] = None,
    extra_paths: Optional[Sequence[Dict[str, Any]]] = None,
    scenario_title: str = "",
    save_path: Union[str, Path] = os.path.join(SCRIPT_DIR, "figure", "planning_solution.png"),
    map_mode: Optional[str] = None,
    dpi: int = 200,
    figsize: Tuple[float, float] = (12, 10),
    suptitle: Optional[str] = None,
) -> Path:
    """Render a single map figure and save to ``save_path``.

    - ``best_path``: primary solution (red by default).
    - ``initial_path``: optional baseline (blue).
    - ``extra_paths``: optional list of dicts with keys ``path``, ``color``, ``label``, etc.
    """
    gmap = grid_map if grid_map is not None else load_routing_grid_map()
    mode = map_mode if map_mode is not None else _default_map_mode()

    specs: List[Dict[str, Any]] = []
    if initial_path is not None and len(_normalize_path(initial_path)) >= 2:
        specs.append(
            {
                "path": initial_path,
                "color": "#2563eb",
                "label": "Initial path",
                "linewidth": 2.0,
                "alpha": 0.9,
            }
        )
    specs.append(
        {
            "path": best_path,
            "color": "#dc2626",
            "label": "Optimized path",
            "linewidth": 2.6,
            "alpha": 0.95,
        }
    )
    if extra_paths:
        specs.extend(list(extra_paths))

    fig, ax = plt.subplots(figsize=figsize)
    draw_paths_on_map_axes(ax, gmap, start, end, specs, map_mode=mode)
    ax.set_title("Path on terrain map", fontsize=14, pad=12)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

    title = suptitle
    if title is None:
        parts = []
        if scenario_title:
            parts.append(scenario_title)
        parts.append(f"start={tuple(start)} end={tuple(end)}")
        title = " | ".join(parts)
    fig.suptitle(title, fontsize=15, y=0.98)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out.resolve()


def visualize_from_optimizer_result(
    result: Dict[str, Any],
    save_path: Union[str, Path],
    *,
    grid_map: Optional[np.ndarray] = None,
    map_mode: Optional[str] = None,
    scenario_title: str = "",
    include_initial: bool = True,
    dpi: int = 200,
) -> Path:
    """Build a figure from a ``run_optimizer``-style dict.

    Expected keys: ``start``, ``end``, ``best_path``. Optional: ``initial_path``.
    """
    start = result.get("start")
    end = result.get("end")
    best_path = result.get("best_path")
    if start is None or end is None:
        raise ValueError("result must include 'start' and 'end' (tuple grid coordinates).")
    if best_path is None:
        raise ValueError("result must include 'best_path'.")
    init = result.get("initial_path") if include_initial else None
    return visualize_planning_solution(
        start=tuple(map(int, start)),
        end=tuple(map(int, end)),
        best_path=best_path,
        grid_map=grid_map,
        initial_path=init,
        scenario_title=scenario_title,
        save_path=save_path,
        map_mode=map_mode,
        dpi=dpi,
    )

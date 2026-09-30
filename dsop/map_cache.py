# -*- coding: utf-8 -*-
"""
dsop/map_cache.py
─────────────────
Lazy-loaded map and derived rasters (distance transform, gradients).
All heavy I/O is performed once and cached at module level.

Public API
----------
_get_cached_map()           -> np.ndarray  raw terrain code array
_get_distance_map()         -> np.ndarray  distance transform from routing obstacles
_get_safety_clearance_distance_map()
                           -> np.ndarray  distance transform from safety hazards
_get_building_distance_map()-> np.ndarray  backward-compatible alias
_get_distance_gradients()   -> (gx, gy)    gradient of distance map
reload_map()                -> np.ndarray  clear caches and reload the active map

grid_map                    module-level alias returned by _get_cached_map()

Constraint helpers (small, used everywhere)
-------------------------------------------
_constraint_violation(G)    -> np.ndarray  sum of max(g, 0) per row
_is_feasible_from_G(G)      -> np.ndarray[bool]
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Optional

from dsop.constants import MAP_MODE, MAP_PATH, ROUTING_MAP_PATH

# ─────────────────────────────────────────────
# Internal caches (module-level singletons)
# ─────────────────────────────────────────────

_CACHED_MAP = None
_DISTANCE_MAP = None
_SAFETY_CLEARANCE_DISTANCE_MAP = None
_DISTANCE_GRADIENT_GX = None
_DISTANCE_GRADIENT_GY = None

_COARSE_OBSTACLE_CACHE: dict = {}


# ─────────────────────────────────────────────
# Map loading
# ─────────────────────────────────────────────

def _get_cached_map() -> np.ndarray:
    global _CACHED_MAP
    if _CACHED_MAP is None:
        path = ROUTING_MAP_PATH if MAP_MODE == "routing_coarse" else MAP_PATH
        _CACHED_MAP = np.load(path, allow_pickle=True)
    return _CACHED_MAP


def _get_distance_map() -> np.ndarray:
    global _DISTANCE_MAP
    if _DISTANCE_MAP is None:
        gm = _get_cached_map()
        if MAP_MODE == "routing_coarse":
            obstacle = (gm == 2) | (gm == 3) | (gm == 5)
        else:
            obstacle = gm == 100
        _DISTANCE_MAP = distance_transform_edt(~obstacle)
    return _DISTANCE_MAP


def _get_safety_clearance_distance_map() -> np.ndarray:
    """Distance transform from the nearest safety hazard.

    In ``routing_coarse`` mode, hazards are buildings (2), roads (3), and
    water (5). In ``mask_gray`` mode, the legacy resident code 100 remains the
    safety hazard.
    """
    global _SAFETY_CLEARANCE_DISTANCE_MAP
    if _SAFETY_CLEARANCE_DISTANCE_MAP is None:
        gm = _get_cached_map()
        if MAP_MODE == "routing_coarse":
            obstacle = (gm == 2) | (gm == 3) | (gm == 5)
        else:
            obstacle = gm == 100
        _SAFETY_CLEARANCE_DISTANCE_MAP = distance_transform_edt(~obstacle)
    return _SAFETY_CLEARANCE_DISTANCE_MAP


def _get_building_distance_map() -> np.ndarray:
    """Backward-compatible alias for the safety clearance distance map."""
    return _get_safety_clearance_distance_map()


def _get_distance_gradients():
    """Return (gx, gy) of the obstacle distance map, computed once and cached."""
    global _DISTANCE_GRADIENT_GX, _DISTANCE_GRADIENT_GY
    if _DISTANCE_GRADIENT_GX is None:
        dm = _get_distance_map().astype(np.float64)
        _DISTANCE_GRADIENT_GY, _DISTANCE_GRADIENT_GX = np.gradient(dm)
    return _DISTANCE_GRADIENT_GX, _DISTANCE_GRADIENT_GY


def reload_map(routing_map_path: Optional[str] = None) -> np.ndarray:
    """Reload the active routing map and clear all dependent caches.

    Use this only when a new routing map was generated inside the same Python
    process. The common build -> restart -> optimize workflow does not need it.
    """
    import sys

    import dsop.constants as _constants
    import dsop.path_metrics as _path_metrics
    import dsop.path_planners as _path_planners

    global _CACHED_MAP, _DISTANCE_MAP, _SAFETY_CLEARANCE_DISTANCE_MAP
    global _DISTANCE_GRADIENT_GX, _DISTANCE_GRADIENT_GY, ROUTING_MAP_PATH, grid_map

    if _constants.MAP_MODE == "routing_coarse":
        resolved = routing_map_path or _constants._resolve_routing_map_path(_constants.PROJECT_DIR)
        _constants.ROUTING_MAP_PATH = resolved
        ROUTING_MAP_PATH = resolved

    _CACHED_MAP = None
    _DISTANCE_MAP = None
    _SAFETY_CLEARANCE_DISTANCE_MAP = None
    _DISTANCE_GRADIENT_GX = None
    _DISTANCE_GRADIENT_GY = None
    _COARSE_OBSTACLE_CACHE.clear()

    _path_metrics._PATH_METRICS_CACHE = {}
    _path_planners._COARSE_TERRAIN_COST_CACHE.clear()
    _path_planners._ASTAR_PATH_CACHE.clear()
    _path_planners._EDGE_COLLISION_CACHE.clear()

    grid_map = _get_cached_map()

    dsop_mod = sys.modules.get("dsop")
    if dsop_mod is not None:
        setattr(dsop_mod, "ROUTING_MAP_PATH", _constants.ROUTING_MAP_PATH)
        setattr(dsop_mod, "grid_map", grid_map)

    return grid_map


# Module-level convenience alias (mirrors original trainingDSOP.grid_map).
grid_map = _get_cached_map()

# ─────────────────────────────────────────────
# Constraint helpers
# ─────────────────────────────────────────────

def _constraint_violation(G: Optional[np.ndarray]) -> np.ndarray:
    """Sum of constraint violations per row: sum(max(g_i, 0)) for each solution."""
    if G is None:
        return np.zeros(0, dtype=float)
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G[None, :]
    return np.sum(np.maximum(G, 0.0), axis=1)


def _is_feasible_from_G(G: Optional[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    """Return bool array: True where total constraint violation <= tol."""
    if G is None:
        return np.array([], dtype=bool)
    return _constraint_violation(G) <= tol

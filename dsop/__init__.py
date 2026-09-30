# -*- coding: utf-8 -*-
"""
dsop  —  DSOP route-planning sub-package
─────────────────────────────────────────
Re-exports the public DSOP API so callers can use ``from dsop import X``.

Module hierarchy (no circular imports):
  constants → (root)
  perf_log  → constants
  map_cache → constants
  preference→ constants, map_cache
  path_metrics → constants, map_cache
  path_planners→ constants, map_cache, path_metrics
  path_operators → constants, map_cache, path_metrics, preference, path_planners
  population → path_operators, path_planners, path_metrics, preference, constants
  representative → path_metrics, preference, constants
  optimizer  → population, representative, path_planners, preference, perf_log
"""

# ── Constants & paths ────────────────────────────────────────────────────────
from dsop.constants import (
    MAP_MODE, MAP_PATH, ROUTING_MAP_PATH, FIGURE_DIR, LOG_DIR,
    DEFAULT_N_OBJ, DEFAULT_REF_DIR_PARTITIONS, DEFAULT_POP_SIZE,
    DEFAULT_RAW_OBJ_LABELS, DEFAULT_OBJ_LABELS, DEFAULT_FOCUS_NAMES, DEFAULT_WEIGHTS, COST_FOCUSED_WEIGHTS,
    PREFERENCE_MAXIMIZE_IDXS,
    _resolve_routing_map_path,
    _neutral_prefer_safety,
    MIN_TOWERS, MAX_TOWERS,
    LINE_SAMPLE_SPACING, LINE_COST_WEIGHT, LINE_LANDUSE_REF_LENGTH,
    ASTAR_TERRAIN_COST_SCALE, SAFETY_CLEARANCE_REF,
    SAFETY_CLEARANCE_WEIGHT, SAFETY_HUMAN_EXPOSURE_WEIGHT,
    BEND_COST_FREE_ANGLE, BEND_COST_SHARP_ANGLE,
    ASTAR_INIT_CELL_SIZES,
    RRT_CELL_SIZE, RRT_GOAL_BIAS, RRT_MAX_ITERS, RRT_MIN_STEP, RRT_MAX_STEP,
    PATH_CROSSOVER_PROB, PATH_MUTATION_PROB, PATH_CROSSOVER_MAX_PAIRS, PATH_MUTATION_TRIES,
    INSGA_STAGNATION_LIMIT, INSGA_GLOBAL_SEARCH_FRACTION, INSGA_LOCAL_SEARCH_FRACTION,
    LOCAL_SEARCH_INTERVAL, OFFSPRING_EMBEDDED_LOCAL_SEARCH,
    LOCAL_SEARCH_INCLUDE_REST_OF_ND_FRONT,
    REPRO_RANDOM_SEED, REPRO_FIXED_START, REPRO_FIXED_END,
    RESIDENT_CODES_MASK_GRAY, RESIDENT_CODES_ROUTING,
    HUMAN_EXPOSURE_CODES_MASK_GRAY, HUMAN_EXPOSURE_CODES_ROUTING,
    TERRAIN_SITING_COST, LINE_CROSSING_COST,
    ROUTE_TERRAIN_SITING_COST, ROUTE_LINE_CROSSING_COST, ROUTE_LINE_TERRAIN_COST,
    _TERRAIN_SITING_COST_ARR, _LINE_CROSSING_COST_ARR, _LINE_TERRAIN_COST_ARR,
    _LINE_TERRAIN_BASE_COST,
    RESIDENT_CODES, HUMAN_EXPOSURE_CODES,
    _maximize_objective_indices, _flip_higher_is_better_columns,
    _build_cost_arrays,
)

# ── Performance logging ───────────────────────────────────────────────────────
from dsop.perf_log import _perf_logging_enabled, _PerfSession, _PerfSpan

# ── Map caches & constraint helpers ──────────────────────────────────────────
from dsop.map_cache import (
    _get_cached_map, _get_distance_map, _get_safety_clearance_distance_map,
    _get_building_distance_map, _get_distance_gradients,
    reload_map,
    _constraint_violation, _is_feasible_from_G,
    grid_map,
)

# ── Preference & utility ──────────────────────────────────────────────────────
from dsop.preference import (
    PreferenceGuidance, _UtilityModelAdapter,
    _guidance_has_utility_model,
    _raw_objectives_to_preference_space,
    _utility_model_scores, _candidate_is_better,
    _select_convergence_pool,
)

# ── Path metrics & geometry ───────────────────────────────────────────────────
from dsop.path_metrics import (
    set_reproducible_rng,
    _normalize_weight_vector, _maximize_objective_indices as _maximize_obj_idxs,
    _canonicalize_objective_batch, _normalize_readable_for_scoring,
    _dedup_consecutive_points, _polyline_cumlen, _resample_polyline,
    _sample_polyline_cells_sparse, _project_to_valid_cell,
    _encode_path_to_legacy_row, _row_signature,
    decode_solution_to_path, compute_path_metrics,
)

# ── Path planners (A*, RRT, initialisation) ───────────────────────────────────
from dsop.path_planners import (
    create_random_path,
    _coarsen_obstacle_grid, _build_coarse_terrain_cost_map,
    _astar_on_binary_grid, _decode_coarse_path_to_fine, _multiresolution_astar_path,
    _edge_is_collision_free, _coarse_edge_is_collision_free, _sample_free_point,
    _rrt_path, _build_astar_rrt_population,
)

# ── Path operators ────────────────────────────────────────────────────────────
from dsop.path_operators import (
    _distance_gradient_at, _path_internal_point_clearances, _adaptive_neighborhood_radius,
    _path_is_collision_free, _sanitize_path_candidate, _select_better_path_bundle,
    _segment_risk_score,
    _path_crossover_operator, _path_mutation_operator,
    _shorten_path_operator, _smoothness_path_operator,
    _safety_path_operator, _local_mutation_operator,
    _adaptive_vns_local_search_operator,
)

# ── Population management & NSGA-II loop ─────────────────────────────────────
from dsop.population import (
    _scalarize_metrics, _coerce_constraint_matrix,
    _evaluate_population_with_decoder,
    _select_weighted_best_index, _select_guided_best_index,
    _crowding_distance_like, _refine_legacy_feasible_front,
    _population_preference_scores, _prefer_population_index,
    _deduplicate_population_rows, _compute_population_state,
    _tournament_better_index, _binary_tournament_select,
    _select_population_environment,
    _encode_path_candidate_row, _generate_insga_offspring_row,
    _apply_local_search_population, _update_stagnation_counts,
    _apply_hybrid_global_search_population,
    _record_convergence_snapshot, _prepare_initial_population,
    _run_insga4_core,
)

# ── Representative path selection ────────────────────────────────────────────
from dsop.representative import (
    select_representative_paths,
)

# ── Top-level optimizer API ───────────────────────────────────────────────────
from dsop.optimizer import (
    run_optimizer_continuous, run_optimizer, visualize_solution_result,
)

__all__ = [
    # constants
    "MAP_MODE", "MAP_PATH", "ROUTING_MAP_PATH", "FIGURE_DIR", "LOG_DIR",
    "DEFAULT_N_OBJ", "DEFAULT_REF_DIR_PARTITIONS", "DEFAULT_POP_SIZE",
    "DEFAULT_RAW_OBJ_LABELS", "DEFAULT_OBJ_LABELS", "DEFAULT_FOCUS_NAMES", "DEFAULT_WEIGHTS", "COST_FOCUSED_WEIGHTS",
    "PREFERENCE_MAXIMIZE_IDXS",
    "_resolve_routing_map_path",
    "_neutral_prefer_safety",
    "MIN_TOWERS", "MAX_TOWERS", "RESIDENT_CODES", "HUMAN_EXPOSURE_CODES",
    "REPRO_RANDOM_SEED", "REPRO_FIXED_START", "REPRO_FIXED_END",
    # perf_log
    "_perf_logging_enabled", "_PerfSession", "_PerfSpan",
    # map_cache
    "_get_cached_map", "_get_distance_map", "_get_safety_clearance_distance_map", "_get_building_distance_map",
    "reload_map",
    "_get_distance_gradients", "_constraint_violation", "_is_feasible_from_G",
    "grid_map", "set_reproducible_rng",
    # preference
    "PreferenceGuidance", "_UtilityModelAdapter", "_guidance_has_utility_model",
    "_raw_objectives_to_preference_space", "_utility_model_scores",
    "_candidate_is_better", "_select_convergence_pool",
    # path_metrics
    "_normalize_weight_vector", "_canonicalize_objective_batch",
    "_normalize_readable_for_scoring",
    "_encode_path_to_legacy_row", "_row_signature",
    "decode_solution_to_path", "compute_path_metrics",
    "_sample_polyline_cells_sparse", "_project_to_valid_cell",
    "_resample_polyline", "_dedup_consecutive_points",
    # path_planners
    "create_random_path", "_multiresolution_astar_path",
    "_edge_is_collision_free", "_rrt_path", "_build_astar_rrt_population",
    # path_operators
    "_path_crossover_operator", "_path_mutation_operator",
    "_shorten_path_operator", "_smoothness_path_operator",
    "_safety_path_operator", "_local_mutation_operator",
    "_adaptive_vns_local_search_operator",
    "_sanitize_path_candidate", "_select_better_path_bundle",
    # population
    "_evaluate_population_with_decoder",
    "_select_weighted_best_index", "_select_guided_best_index",
    "_crowding_distance_like", "_refine_legacy_feasible_front",
    "_compute_population_state", "_select_population_environment",
    "_encode_path_candidate_row", "_generate_insga_offspring_row",
    "_apply_local_search_population", "_apply_hybrid_global_search_population",
    "_run_insga4_core",
    # representative
    "select_representative_paths",
    # optimizer
    "run_optimizer", "run_optimizer_continuous", "visualize_solution_result",
]

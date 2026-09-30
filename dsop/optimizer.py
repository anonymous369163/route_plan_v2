# -*- coding: utf-8 -*-
"""
dsop/optimizer.py
─────────────────
Top-level public API for running the INSGA-II-style route-plan optimiser.

Public API
----------
run_optimizer(...)              full fresh-start optimisation with optional plotting
run_optimizer_continuous(...)   stateless continuation API (no plots, warm-start OK)
visualize_solution_result(...)  save a terrain-overview figure from a result dict
"""

import os
import random
import numpy as np
from typing import Optional

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from dsop.constants import (
    MAP_MODE, FIGURE_DIR, DEFAULT_N_OBJ, DEFAULT_POP_SIZE, resolve_pop_size,
    DEFAULT_WEIGHTS, DEFAULT_RAW_OBJ_LABELS,
    MIN_TOWERS, MAX_TOWERS, RESIDENT_CODES,
)
from dsop.perf_log import _perf_logging_enabled, _PerfSession
from dsop.map_cache import _get_cached_map, _is_feasible_from_G
from dsop.path_metrics import (
    compute_path_metrics, decode_solution_to_path,
    _encode_path_to_legacy_row, _normalize_weight_vector,
)
from dsop.evaluation_archive import (
    EvaluationRecorder,
    ObjectiveConsistencyError,
    capture_evaluations,
)
from dsop.path_planners import _build_astar_rrt_population, create_random_path
from dsop.preference import PreferenceGuidance
from dsop.population import (
    _run_insga4_core,
    _select_weighted_best_index, _select_guided_best_index,
    _evaluate_population_with_decoder, _refine_legacy_feasible_front,
    _compute_population_state, _is_pinsga2_official_profile, _is_search_guidance_disabled_profile,
    _is_preference_env_only_profile,
    _is_utility_scalarized_profile,
)

os.makedirs(FIGURE_DIR, exist_ok=True)

# Lazy import: utils.py lives in the project root and pulls in matplotlib/cv2,
# so we defer it to the functions that actually need it.
def _import_utils():
    from utils import (
        ConvergenceCallback as _CC,
        plot_convergence_satisfaction,
        plot_convergence_objectives,
        visualize_path_on_terrain,
        plot_paths_on_map,
        plot_metrics_comparison,
    )
    return _CC, plot_convergence_satisfaction, plot_convergence_objectives, \
           visualize_path_on_terrain, plot_paths_on_map, plot_metrics_comparison


# ─────────────────────────────────────────────
# Continuous (warm-start) optimizer
# ─────────────────────────────────────────────

def run_optimizer_continuous(
    user_weights=None, start=None, end=None, n_gen=50,
    ref_dirs=None, guidance=None,
    prev_pop_X: Optional[np.ndarray] = None,
    perf_log: Optional[bool] = None,
    algorithm_profile: Optional[str] = None,
    vf_dominance_state: Optional[dict] = None,
    generation_observer: Optional[object] = None,
    environment_selector: Optional[object] = None,
    evaluation_recorder: Optional[EvaluationRecorder] = None,
):
    """Run the Phase-B INSGA-II core for ``n_gen`` generations without plots.

    Designed for iterative / interactive workflows where the caller manages
    plotting externally and passes the previous population as warm-start.
    """

    def _select_profile_best_index(
        F_batch,
        G_batch,
        weights,
        *,
        algorithm_profile=None,
        vf_dominance_state=None,
        guidance=None,
    ):
        if _is_pinsga2_official_profile(algorithm_profile):
            objs = np.asarray(F_batch, dtype=float)
            if objs.ndim == 1:
                objs = objs[None, :]
            if len(objs) == 0:
                return None

            G_arr = None if G_batch is None else np.asarray(G_batch, dtype=float)
            if G_arr is not None and G_arr.ndim == 1:
                G_arr = G_arr[None, :]

            if G_arr is not None:
                from dsop.map_cache import _constraint_violation

                cv = _constraint_violation(G_arr)
                is_feasible = cv <= 1e-8
            else:
                cv = np.zeros(len(objs), dtype=float)
                is_feasible = np.ones(len(objs), dtype=bool)

            pool_idx = np.where(is_feasible)[0] if np.any(is_feasible) else np.argsort(cv)[: min(10, len(objs))]
            vf_res = None if vf_dominance_state is None else vf_dominance_state.get("vf_res")
            if vf_res is not None and getattr(vf_res, "vf", None) is not None:
                try:
                    scores = np.asarray(vf_res.vf(objs[pool_idx] * -1.0), dtype=float).reshape(-1)
                except Exception:
                    scores = np.zeros(0, dtype=float)
                if len(scores) == len(pool_idx):
                    scores = np.nan_to_num(scores, nan=1e9, posinf=1e9, neginf=-1e9)
                    return int(pool_idx[int(np.argmin(scores))])
            return _select_weighted_best_index(F_batch, G_batch, weights)

        if _is_preference_env_only_profile(algorithm_profile):
            return _select_weighted_best_index(F_batch, G_batch, weights)

        # ``preference_benchmark_env_only`` disables utility guidance between
        # interaction generations; it does not change the final-output rule.
        # Recommendation is a separate operation and must still select the
        # learned-utility argmin from the cumulative archive when a model is
        # available.  ``_select_guided_best_index`` already falls back to the
        # weighted rule when ``guidance`` is absent.
        return _select_guided_best_index(F_batch, G_batch, weights, guidance=guidance)

    en   = _perf_logging_enabled(perf_log)
    meta = [f"MAP_MODE={MAP_MODE}", f"n_gen={n_gen}", "api=run_optimizer_continuous"]
    with _PerfSession("run_optimizer_continuous", meta, enabled=en) as perf:
        with perf.span("load_map_normalize_weights"):
            grid_map          = _get_cached_map()
            h, w              = grid_map.shape
            selection_weights = _normalize_weight_vector(user_weights, n_obj=DEFAULT_N_OBJ)

        with perf.span("sample_start_end_if_needed"):
            if start is None or end is None:
                while True:
                    start = (random.randint(0, w - 1), random.randint(0, h - 1))
                    end   = (random.randint(0, w - 1), random.randint(0, h - 1))
                    if (
                        start != end
                        and grid_map[start[1], start[0]] not in RESIDENT_CODES
                        and grid_map[end[1], end[0]] not in RESIDENT_CODES
                    ):
                        break

        pop_size = resolve_pop_size(DEFAULT_POP_SIZE)
        perf.note(f"pop_size={pop_size} start={start} end={end}")

        recorder = evaluation_recorder if evaluation_recorder is not None else EvaluationRecorder()
        with perf.span("_run_insga4_core"):
            with capture_evaluations(recorder, source="optimizer_core"):
                core = _run_insga4_core(
                    start=start, end=end, grid_map=grid_map,
                    pop_size=pop_size, n_gen=n_gen,
                    initial_population=prev_pop_X,
                    guidance=guidance, callback=None, perf_session=perf,
                    algorithm_profile=algorithm_profile,
                    vf_dominance_state=vf_dominance_state,
                    generation_observer=generation_observer,
                    environment_selector=environment_selector,
                )
        pop_X_raw = np.asarray(core["pop_X"], dtype=float)
        pop_F_raw = np.asarray(core["pop_F"], dtype=float)
        pop_G_raw = None if core["pop_G"] is None else np.asarray(core["pop_G"], dtype=float)
        last_environment_state = dict(core.get("last_environment_state") or {})

        core_recorded_evaluations = len(recorder.records) > 0
        for idx, row in enumerate(pop_X_raw):
            try:
                recorder.register_snapshot_candidate(
                    path=decode_solution_to_path(row, start, end),
                    objectives=pop_F_raw[idx],
                    constraints=None if pop_G_raw is None else pop_G_raw[idx],
                    source="final_population_snapshot",
                )
            except ObjectiveConsistencyError:
                # Test doubles and external observers can expose multiple rows
                # through a deliberately lossy decoder without performing the
                # underlying evaluations.  Real optimiser evaluations are
                # already recorded and must retain the strict consistency check.
                if core_recorded_evaluations:
                    raise

        with perf.span("feasible_filter_nondominated_sort"):
            pop_feasible = (
                _is_feasible_from_G(pop_G_raw) if pop_G_raw is not None
                else np.ones(len(pop_F_raw), dtype=bool)
            )
            if np.any(pop_feasible):
                feas_idx = np.where(pop_feasible)[0]
                F_feas   = pop_F_raw[feas_idx]
                X_feas   = pop_X_raw[feas_idx]
                G_feas   = None if pop_G_raw is None else pop_G_raw[feas_idx]
            else:
                F_feas, X_feas = pop_F_raw, pop_X_raw
                G_feas = pop_G_raw
            if _is_pinsga2_official_profile(algorithm_profile):
                state = _compute_population_state(
                    F_feas,
                    G_feas,
                    guidance=None,
                    algorithm_profile=algorithm_profile,
                    vf_dominance_state=vf_dominance_state,
                )
                nds = state["nd_idx"]
            else:
                nds = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)

        with perf.span("select_best_from_evaluation_archive"):
            archive = recorder.archive_snapshot()
            archive_F = np.asarray(archive["F"], dtype=float)
            archive_G = np.asarray(archive["G"], dtype=float)
            if len(archive_F) == 0:
                raise RuntimeError("The evaluation archive contains no feasible candidate.")
            evaluated_feasible = recorder.feasible_snapshot()
            scalarized_output = _is_utility_scalarized_profile(algorithm_profile)
            selection_F = (
                np.asarray(evaluated_feasible["F"], dtype=float)
                if scalarized_output
                else archive_F
            )
            selection_G = (
                np.asarray(evaluated_feasible["G"], dtype=float)
                if scalarized_output
                else archive_G
            )
            selection_paths = evaluated_feasible["paths"] if scalarized_output else archive["paths"]
            best_idx = _select_profile_best_index(
                selection_F,
                selection_G,
                selection_weights,
                algorithm_profile=algorithm_profile,
                vf_dominance_state=vf_dominance_state,
                guidance=guidance,
            )
            if best_idx is None:
                raise RuntimeError("Unable to select a best candidate from the evaluation archive.")
            best_path = [tuple(point) for point in selection_paths[best_idx]]
            best_metrics_raw = np.asarray(selection_F[best_idx], dtype=float)
            best_constraints = np.asarray(selection_G[best_idx], dtype=float)
            # Compatibility projection only: the fixed-width legacy encoder
            # may resample a recorded path.  ``best_path`` and its stored
            # objectives are the authoritative recommendation representation.
            best_vars = _encode_path_to_legacy_row(
                best_path,
                grid_map,
                num_towers=len(best_path) - 2,
            )
            with capture_evaluations(recorder, source="recommendation_consistency"):
                check_metrics, check_constraints = compute_path_metrics(best_path, grid_map)
            recorder.assert_consistent(
                path=best_path,
                objectives=check_metrics,
                constraints=check_constraints,
            )
            # ``archive_X`` is likewise a legacy visualization/warm-start
            # projection; archive membership and all audits use
            # ``archive_paths`` paired with ``archive_F``/``archive_G``.
            archive_X = np.asarray(
                [
                    _encode_path_to_legacy_row(path, grid_map, num_towers=len(path) - 2)
                    for path in archive["paths"]
                ],
                dtype=float,
            )

        if perf.path:
            perf.note(f"perf_log_file={perf.path}")

        return {
            "pareto_X":    X_feas[nds],
            "pareto_F":    F_feas[nds],
            "feasible_X":  X_feas,
            "feasible_F":  F_feas,
            "pop_X":       pop_X_raw,
            "pop_F":       pop_F_raw,
            "archive_X":   archive_X,
            "archive_F":   archive_F,
            "archive_G":   archive_G,
            "archive_paths": archive["paths"],
            "archive_sources": archive["sources"],
            "evaluated_feasible_F": evaluated_feasible["F"],
            "evaluated_feasible_G": evaluated_feasible["G"],
            "evaluated_feasible_paths": evaluated_feasible["paths"],
            "evaluated_feasible_sources": evaluated_feasible["sources"],
            "archive_stats": {
                "n_evaluations": archive["n_evaluations"],
                "n_unique_decisions": archive["n_unique_decisions"],
                "n_feasible_decisions": archive["n_feasible_decisions"],
                "n_archive": len(archive_F),
                "source_counts": dict(archive["source_counts"]),
            },
            "best_vars":   best_vars,
            "best_metrics": best_metrics_raw,
            "best_constraints": best_constraints,
            "recommendation_source": (
                "evaluated_feasible_utility_objective"
                if scalarized_output
                else "evaluation_archive"
            ),
            "finite_run_nondominance_guaranteed": not scalarized_output,
            "best_path":   best_path,
            "start":       tuple(map(int, start)),
            "end":         tuple(map(int, end)),
            "encoding":    "legacy_absolute",
            "n_var":       int(1 + 2 * MAX_TOWERS),
            "query_X":     last_environment_state.get("query_X"),
            "query_F":     last_environment_state.get("query_F"),
            "query_G":     last_environment_state.get("query_G"),
            "query_pool_source": last_environment_state.get("query_pool_source"),
            "query_pool_generation": last_environment_state.get("query_pool_generation"),
            "selection_context_X": last_environment_state.get("selection_context_X"),
            "selection_context_F": last_environment_state.get("selection_context_F"),
            "selection_context_G": last_environment_state.get("selection_context_G"),
            "perf_log_path": perf.path or None,
        }


# ─────────────────────────────────────────────
# Legacy full-run optimizer
# ─────────────────────────────────────────────

def run_optimizer(
    user_weights=None, initial_population: np.ndarray = None,
    start=None, end=None, n_gen=50,
    ref_dirs: np.ndarray = None, guidance: PreferenceGuidance = None,
    skip_plots: bool = True, init_strategy: str = "astar_rrt",
    use_elite_local_search: bool = True,
    perf_log: Optional[bool] = None,
    repro_seed_applied: Optional[int] = None,
    figure_dir: Optional[str] = None,
):
    """Full optimisation run (fresh start, optional plotting, optional elite local search).

    Parameters
    ----------
    user_weights        Objective weights for selecting the *best* candidate to display.
    initial_population  Optional warm-start population (numpy array of chromosome rows).
    start / end         (x, y) grid coordinates; sampled randomly if None.
    n_gen               Number of NSGA-II generations.
    guidance            PreferenceGuidance with optional utility_model.
    skip_plots          If False, emit comparison / convergence figures.
    init_strategy       "astar_rrt" (default) or "random".
    use_elite_local_search  Apply local-search refinement to the returned Pareto front.
    repro_seed_applied  Seed value logged in perf log for reproducibility annotation.
    figure_dir          Directory for ``skip_plots=False`` outputs (default: ``FIGURE_DIR``).
    """
    plot_dir = figure_dir if figure_dir is not None else FIGURE_DIR
    en   = _perf_logging_enabled(perf_log)
    meta = [
        f"MAP_MODE={MAP_MODE}", f"n_gen={n_gen}",
        f"skip_plots={skip_plots}", f"init_strategy={init_strategy}",
        f"use_elite_local_search={use_elite_local_search}",
        "api=run_optimizer",
        f"run_start_arg={start}", f"run_end_arg={end}",
        f"figure_dir={plot_dir}",
    ]
    if repro_seed_applied is not None:
        meta.append(f"repro_seed_applied={int(repro_seed_applied)}")

    with _PerfSession("run_optimizer", meta, enabled=en) as perf:
        with perf.span("load_map_normalize_weights"):
            grid_map          = _get_cached_map()
            h, w              = grid_map.shape
            selection_weights = _normalize_weight_vector(user_weights, n_obj=DEFAULT_N_OBJ)

        with perf.span("sample_start_end_if_needed"):
            if start is None or end is None:
                while True:
                    start = (random.randint(0, w - 1), random.randint(0, h - 1))
                    end   = (random.randint(0, w - 1), random.randint(0, h - 1))
                    if (
                        start != end
                        and grid_map[start[1], start[0]] not in RESIDENT_CODES
                        and grid_map[end[1], end[0]] not in RESIDENT_CODES
                    ):
                        break

        recorder = EvaluationRecorder()
        with perf.span("initial_random_path_and_metrics"):
            num_towers      = random.randint(MIN_TOWERS, MAX_TOWERS)
            initial_path    = create_random_path(start, end, num_towers, grid_map)
            with capture_evaluations(recorder, source="initial_candidate"):
                initial_metrics, _ = compute_path_metrics(initial_path, grid_map)
            decode_legacy   = lambda row: decode_solution_to_path(row, start, end)

        pop_size = int(len(ref_dirs)) if ref_dirs is not None else resolve_pop_size(DEFAULT_POP_SIZE)
        perf.note(f"pop_size={pop_size} start={start} end={end}")

        init_source   = str(init_strategy)
        init_pop_arr  = None
        with perf.span("build_init_population"):
            if initial_population is not None:
                pop_arr      = np.asarray(initial_population, dtype=float)
                init_source  = "external"
                init_pop_arr = np.asarray(pop_arr, dtype=float).copy()
            elif init_strategy == "astar_rrt":
                pop_arr      = _build_astar_rrt_population(start, end, grid_map, pop_size=pop_size)
                init_pop_arr = np.asarray(pop_arr, dtype=float).copy()

        init_pop_F = None
        init_pop_G = None
        with perf.span("evaluate_init_population_pick_display_path"):
            if init_pop_arr is not None and len(init_pop_arr) > 0:
                with capture_evaluations(recorder, source="initial_population"):
                    init_pop_F, init_pop_G = _evaluate_population_with_decoder(
                        init_pop_arr, decode_legacy, grid_map
                    )
                init_best_idx = _select_weighted_best_index(init_pop_F, init_pop_G, selection_weights)
                if init_best_idx is not None:
                    initial_path    = decode_legacy(init_pop_arr[init_best_idx])
                    initial_metrics = np.asarray(init_pop_F[init_best_idx], dtype=float)

        conv_cb = None
        if not skip_plots:
            _CC, *_ = _import_utils()
            conv_cb = _CC()

        with perf.span("_run_insga4_core"):
            with capture_evaluations(recorder, source="optimizer_core"):
                core = _run_insga4_core(
                    start=start, end=end, grid_map=grid_map,
                    pop_size=pop_size, n_gen=n_gen,
                    initial_population=init_pop_arr,
                    guidance=guidance, callback=conv_cb, perf_session=perf,
                )
        pop_F_raw = np.asarray(core["pop_F"], dtype=float)
        pop_X_raw = np.asarray(core["pop_X"], dtype=float)
        pop_G_raw = None if core["pop_G"] is None else np.asarray(core["pop_G"], dtype=float)

        core_recorded_evaluations = len(recorder.records) > 0
        for idx, row in enumerate(pop_X_raw):
            try:
                recorder.register_snapshot_candidate(
                    path=decode_legacy(row),
                    objectives=pop_F_raw[idx],
                    constraints=None if pop_G_raw is None else pop_G_raw[idx],
                    source="final_population_snapshot",
                )
            except ObjectiveConsistencyError:
                if core_recorded_evaluations:
                    raise

        with perf.span("feasible_filter_nondominated_front"):
            pop_feasible = (
                _is_feasible_from_G(pop_G_raw) if pop_G_raw is not None
                else np.ones(len(pop_F_raw), dtype=bool)
            )
            if np.any(pop_feasible):
                feas_idx = np.where(pop_feasible)[0]
                F_feas   = pop_F_raw[feas_idx]
                X_feas   = pop_X_raw[feas_idx]
            else:
                F_feas   = pop_F_raw
                X_feas   = pop_X_raw
            nds_local = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
            true_pf   = F_feas[nds_local]
            pareto_X  = X_feas[nds_local]

        with perf.span("refine_legacy_feasible_front"):
            refined_X = np.asarray(pareto_X, dtype=float)
            refined_F = np.asarray(true_pf, dtype=float)
            if use_elite_local_search and len(pareto_X) > 0:
                with capture_evaluations(recorder, source="post_loop_refinement"):
                    refined_X, refined_F = _refine_legacy_feasible_front(
                        pareto_X, true_pf, start=start, end=end, grid_map=grid_map,
                        max_refine=min(12, max(4, len(pareto_X))),
                    )

        with perf.span("select_best_from_evaluation_archive"):
            archive = recorder.archive_snapshot()
            archive_F = np.asarray(archive["F"], dtype=float)
            archive_G = np.asarray(archive["G"], dtype=float)
            if len(archive_F) == 0:
                raise RuntimeError("The evaluation archive contains no feasible candidate.")
            best_idx = _select_guided_best_index(
                archive_F,
                archive_G,
                selection_weights,
                guidance=guidance,
            )
            if best_idx is None:
                raise RuntimeError("Unable to select a best candidate from the evaluation archive.")
            best_path = [tuple(point) for point in archive["paths"][best_idx]]
            best_objs = np.asarray(archive_F[best_idx], dtype=float)
            best_G = np.asarray(archive_G[best_idx], dtype=float)
            # Compatibility projection only; see the continuous optimiser
            # branch above.  Never use this lossy row for archive membership.
            best_vars = _encode_path_to_legacy_row(
                best_path,
                grid_map,
                num_towers=len(best_path) - 2,
            )
            with capture_evaluations(recorder, source="recommendation_consistency"):
                check_metrics, check_constraints = compute_path_metrics(best_path, grid_map)
            recorder.assert_consistent(
                path=best_path,
                objectives=check_metrics,
                constraints=check_constraints,
            )
            # Exact decision truth remains ``archive_paths``.  This fixed-row
            # projection is retained only for legacy callers.
            archive_X = np.asarray(
                [
                    _encode_path_to_legacy_row(path, grid_map, num_towers=len(path) - 2)
                    for path in archive["paths"]
                ],
                dtype=float,
            )
            evaluated_feasible = recorder.feasible_snapshot()

        best_objs = np.asarray(best_objs, dtype=float)

        if not skip_plots:
            os.makedirs(plot_dir, exist_ok=True)
            _CC, plot_conv_sat, plot_conv_obj, viz_terrain, _, plot_metrics_cmp = _import_utils()
            with perf.span("plot_metrics_comparison"):
                plot_metrics_cmp(
                    initial_metrics, best_objs,
                    labels=list(DEFAULT_RAW_OBJ_LABELS),
                    save_path=os.path.join(plot_dir, "comparison.png"),
                )
            with perf.span("visualize_path_on_terrain"):
                viz_terrain(
                    grid_map, best_path, start, end,
                    save_path=os.path.join(plot_dir, "path_on_terrain.png"),
                    initial_path=initial_path,
                )
            with perf.span("plot_convergence_satisfaction"):
                plot_conv_sat(
                    history=core.get("history"), weights=selection_weights,
                    save_path=os.path.join(plot_dir, "convergence_satisfaction.png"),
                    callback=conv_cb,
                )
            with perf.span("plot_convergence_objectives"):
                plot_conv_obj(
                    history=core.get("history"), weights=selection_weights,
                    save_path=os.path.join(plot_dir, "convergence_objectives.png"),
                    callback=conv_cb,
                )

        if perf.path:
            perf.note(f"perf_log_file={perf.path}")

        return {
            "pareto_X":    pareto_X,
            "pareto_F":    true_pf,
            "post_refinement_X": refined_X,
            "post_refinement_F": refined_F,
            "feasible_X":  X_feas,
            "feasible_F":  F_feas,
            "pop_X":       pop_X_raw,
            "pop_F":       pop_F_raw,
            "archive_X":   archive_X,
            "archive_F":   archive_F,
            "archive_G":   archive_G,
            "archive_paths": archive["paths"],
            "archive_sources": archive["sources"],
            "evaluated_feasible_F": evaluated_feasible["F"],
            "evaluated_feasible_G": evaluated_feasible["G"],
            "evaluated_feasible_paths": evaluated_feasible["paths"],
            "evaluated_feasible_sources": evaluated_feasible["sources"],
            "archive_stats": {
                "n_evaluations": archive["n_evaluations"],
                "n_unique_decisions": archive["n_unique_decisions"],
                "n_feasible_decisions": archive["n_feasible_decisions"],
                "n_archive": len(archive_F),
                "source_counts": dict(archive["source_counts"]),
            },
            "best_vars":   best_vars,
            "best_metrics": best_objs,
            "best_constraints": best_G,
            "recommendation_source": "evaluation_archive",
            "finite_run_nondominance_guaranteed": True,
            "best_path":   best_path,
            "start":       tuple(map(int, start)),
            "end":         tuple(map(int, end)),
            "initial_path": list(initial_path),
            "encoding":    "legacy_absolute",
            "n_var":       int(1 + 2 * MAX_TOWERS),
            "init_strategy": init_source,
            "perf_log_path": perf.path or None,
        }


# ─────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────

def visualize_solution_result(
    result: dict,
    save_path: Optional[str] = None,
    *,
    scenario_title: str = "",
    include_initial: bool = True,
    dpi: int = 200,
) -> str:
    """Save a terrain overview figure from a ``run_optimizer`` / ``run_optimizer_continuous`` result.

    Requires the result to contain ``start``, ``end``, ``best_path``.
    Optional ``initial_path`` is used when present (``run_optimizer`` provides it).
    """
    from scenario_visualization import visualize_from_optimizer_result
    out  = save_path or os.path.join(FIGURE_DIR, "solution_overview.png")
    path = visualize_from_optimizer_result(
        result, out,
        scenario_title=scenario_title,
        include_initial=include_initial,
        map_mode=MAP_MODE,
        dpi=int(dpi),
    )
    return str(path)

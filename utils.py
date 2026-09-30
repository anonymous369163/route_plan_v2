"""
Plotting and visualization utilities. No impact on algorithm performance.
Extracted from the legacy DSOP training module for separation of concerns.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Optional, Sequence, Tuple

from pymoo.core.callback import Callback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figure")
os.makedirs(FIGURE_DIR, exist_ok=True)
DEFAULT_RAW_OBJ_LABELS = ["cost", "length", "safety_penalty", "bend_penalty"]
DEFAULT_OBJ_LABELS = ["cost", "length", "safety", "smoothness"]
PREFERENCE_MAXIMIZE_IDXS = (2, 3)

# Terrain colors for path visualization
TERRAIN_COLORS_MASK_GRAY = {
    9: (0.55, 0.55, 0.55),
    27: (0.18, 0.62, 0.28),
    36: (0.85, 0.46, 0.18),
    91: (0.88, 0.81, 0.56),
    100: (0.45, 0.25, 0.08),
    118: (0.30, 0.30, 0.32),
    255: (0.92, 0.92, 0.92),
}
TERRAIN_COLORS_ROUTING = {
    0: (0.40, 0.40, 0.40),
    1: (0.95, 0.82, 0.42),
    2: (0.85, 0.11, 0.38),
    3: (1.0, 0.30, 0.30),
    4: (0.18, 0.80, 0.44),
    5: (0.12, 0.37, 0.82),
}


# ─────────────────────────────────────────────
# Convergence callback (collects F,G per gen for plotting)
# ─────────────────────────────────────────────

class ConvergenceCallback(Callback):
    """Collect F, G per generation for convergence plot. More reliable than result.history."""

    def __init__(self):
        super().__init__()
        self.F_list: List[np.ndarray] = []
        self.G_list: List[Optional[np.ndarray]] = []

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")
        self.F_list.append(np.asarray(F, dtype=float).copy() if F is not None else np.array([]))
        self.G_list.append(np.asarray(G, dtype=float).copy() if G is not None else None)


def _extract_FG_from_history(history) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Extract (F, G) from pymoo history (list of algorithm snapshots)."""
    out = []
    for entry in history:
        pop = getattr(entry, "pop", None)
        if pop is None:
            continue
        F = pop.get("F") if hasattr(pop, "get") else None
        G = pop.get("G") if pop is not None and F is not None else None
        if F is not None:
            out.append((np.asarray(F, dtype=float), np.asarray(G, dtype=float) if G is not None else None))
    return out


def _get_convergence_generation_data(
    history=None,
    callback: Optional[ConvergenceCallback] = None,
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    if callback is not None and len(callback.F_list) > 0:
        return [(callback.F_list[i], callback.G_list[i]) for i in range(len(callback.F_list))]
    if history is not None and len(history) > 0:
        return _extract_FG_from_history(history)
    return []


def _accumulate_best_series(series: Sequence[float], maximize: bool = False) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    best = None
    for i, val in enumerate(arr):
        if np.isnan(val):
            out[i] = np.nan if best is None else best
            continue
        if best is None:
            best = float(val)
        else:
            best = max(best, float(val)) if maximize else min(best, float(val))
        out[i] = best
    return out


def _compute_convergence_statistics(
    history=None,
    weights: Sequence[float] = None,
    callback: Optional[ConvergenceCallback] = None,
    *,
    _training_module=None,
) -> Optional[dict]:
    """Compute convergence stats for plotting. Uses lazy import to avoid circular dependency."""
    if _training_module is None:
        import dsop as _training_module

    gen_data = _get_convergence_generation_data(history=history, callback=callback)
    if not gen_data:
        return None

    _select_convergence_pool = _training_module._select_convergence_pool
    _canonicalize_objective_batch = _training_module._canonicalize_objective_batch
    _normalize_weight_vector = _training_module._normalize_weight_vector
    _maximize_objective_indices = _training_module._maximize_objective_indices
    _flip_higher_is_better_columns = _training_module._flip_higher_is_better_columns

    all_readable = []
    for F, G in gen_data:
        if F is None or len(F) == 0:
            continue
        pool_F = _select_convergence_pool(F, G)
        if pool_F is None or len(pool_F) == 0:
            continue
        all_readable.append(_canonicalize_objective_batch(pool_F))

    if not all_readable:
        return None

    global_F = np.vstack(all_readable)
    n_obj = int(global_F.shape[1])
    w = _normalize_weight_vector(weights, n_obj=n_obj)
    g_mins = global_F.min(axis=0)
    g_maxs = global_F.max(axis=0)
    g_ranges = g_maxs - g_mins + 1e-8

    best_scores = []
    objective_best = []
    for F, G in gen_data:
        row = np.full(n_obj, np.nan, dtype=float)
        if F is None or len(F) == 0:
            best_scores.append(np.nan)
            objective_best.append(row)
            continue

        pool_F = _select_convergence_pool(F, G)
        if pool_F is None or len(pool_F) == 0:
            best_scores.append(np.nan)
            objective_best.append(row)
            continue

        readable = _canonicalize_objective_batch(pool_F)
        maximize_idxs = set(_maximize_objective_indices(n_obj))
        for j in range(n_obj):
            if j in maximize_idxs:
                row[j] = float(np.max(readable[:, j]))
            else:
                row[j] = float(np.min(readable[:, j]))
        objective_best.append(row)

        normed = (readable - g_mins) / g_ranges
        normed = _flip_higher_is_better_columns(normed)
        best_scores.append(float(np.min(normed.dot(w))))

    best_scores = np.asarray(best_scores, dtype=float)
    objective_best = np.asarray(objective_best, dtype=float)

    valid = ~np.isnan(best_scores)
    if np.any(valid):
        s_min = np.min(best_scores[valid])
        s_max = np.max(best_scores[valid])
        satisfaction = 1.0 - (best_scores - s_min) / (s_max - s_min + 1e-8)
        satisfaction[~valid] = np.nan
    else:
        satisfaction = np.full_like(best_scores, np.nan)

    maximize_idxs = set(_maximize_objective_indices(n_obj))
    objective_best_so_far = np.column_stack([
        _accumulate_best_series(objective_best[:, j], maximize=(j in maximize_idxs))
        for j in range(n_obj)
    ])

    labels = list(DEFAULT_OBJ_LABELS[:n_obj])
    ylabels = []
    for j, label in enumerate(labels):
        if j in maximize_idxs:
            ylabels.append(f"{label} (higher better)")
        else:
            ylabels.append(f"{label} (lower better)")

    return {
        "objective_best": objective_best,
        "objective_best_so_far": objective_best_so_far,
        "satisfaction": satisfaction,
        "labels": labels,
        "ylabels": ylabels,
        "generations": np.arange(len(best_scores)),
    }


def plot_convergence_satisfaction(
    history=None,
    weights: Sequence[float] = None,
    save_path: Optional[str] = None,
    callback: Optional[ConvergenceCallback] = None,
    figure_dir: Optional[str] = None,
) -> None:
    """Plot satisfaction (0-1) over generations. Higher = better preference alignment."""
    stats = _compute_convergence_statistics(history=history, weights=weights, callback=callback)
    if stats is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stats["generations"], stats["satisfaction"], color="#2563eb", linewidth=2, label="Satisfaction")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Satisfaction (0-1)")
    ax.set_title("Convergence: User-weighted Satisfaction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = save_path or os.path.join(figure_dir or FIGURE_DIR, "convergence_satisfaction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_convergence_objectives(
    history=None,
    weights: Sequence[float] = None,
    save_path: Optional[str] = None,
    callback: Optional[ConvergenceCallback] = None,
    figure_dir: Optional[str] = None,
) -> None:
    """Plot convergence of each objective using feasible-population best and best-so-far."""
    stats = _compute_convergence_statistics(history=history, weights=weights, callback=callback)
    if stats is None:
        return

    labels = stats["labels"]
    ylabels = stats["ylabels"]
    gens = stats["generations"]
    current = stats["objective_best"]
    best_so_far = stats["objective_best_so_far"]

    n_obj = len(labels)
    n_cols = 2
    n_rows = int(np.ceil(n_obj / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.ravel()
    colors = ["#dc2626", "#2563eb", "#059669", "#d97706", "#7c3aed", "#0891b2"]

    for j in range(n_obj):
        ax = axes_flat[j]
        color = colors[j % len(colors)]
        ax.plot(gens, current[:, j], color=color, alpha=0.35, linewidth=1.5, label="Best in generation")
        ax.plot(gens, best_so_far[:, j], color=color, linewidth=2.2, label="Best so far")
        ax.set_title(labels[j])
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabels[j])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for ax in axes_flat[n_obj:]:
        ax.axis("off")

    fig.suptitle("Convergence: Objective Trajectories", fontsize=14, y=0.98)
    plt.tight_layout()
    path = save_path or os.path.join(figure_dir or FIGURE_DIR, "convergence_objectives.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Path visualization
# ─────────────────────────────────────────────

def build_terrain_cmap_for_map(grid_map: np.ndarray, map_mode: str = "routing_coarse") -> ListedColormap:
    """Build colormap for terrain visualization based on map_mode."""
    cmap_list = [(0.08, 0.08, 0.08)] * 256
    colors = TERRAIN_COLORS_ROUTING if map_mode == "routing_coarse" else TERRAIN_COLORS_MASK_GRAY
    for code, rgb in colors.items():
        if 0 <= code < 256:
            cmap_list[code] = rgb
    return ListedColormap(cmap_list)


def visualize_path_on_terrain(
    grid_map: np.ndarray,
    path: Sequence[tuple],
    start: tuple,
    end: tuple,
    save_path: Optional[str] = None,
    initial_path: Optional[Sequence[tuple]] = None,
    figure_dir: Optional[str] = None,
    map_mode: str = "routing_coarse",
) -> None:
    """Draw planned path on terrain-colored map for quick debugging."""
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = build_terrain_cmap_for_map(grid_map, map_mode=map_mode)
    ax.imshow(
        grid_map,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=255,
    )
    ax.scatter(start[0], start[1], s=400, color="#00ff00", marker="o", edgecolor="white", linewidth=2.5, label="Start", zorder=10)
    ax.scatter(end[0], end[1], s=500, color="#ff0000", marker="*", edgecolor="white", linewidth=2.0, label="End", zorder=10)

    if initial_path is not None and len(initial_path) > 0:
        ix, iy = zip(*initial_path)
        ax.plot(ix, iy, color="#2563eb", linewidth=2.0, alpha=0.9, label="Initial")
    if path is not None and len(path) > 0:
        px, py = zip(*path)
        ax.plot(px, py, color="#dc2626", linewidth=2.5, alpha=0.95, label="Optimized path")

    ax.set_title("Path on terrain map", fontsize=14, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    plt.tight_layout()
    path_out = save_path or os.path.join(figure_dir or FIGURE_DIR, "path_on_terrain.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_paths_on_map(
    grid_map,
    user_path=None,
    default_path=None,
    start=None,
    end=None,
    save_path=None,
    figure_dir: Optional[str] = None,
):
    """Plot user path and default path on gray map."""
    plt.figure(figsize=(10, 10))
    cmap = plt.cm.get_cmap('gray')
    plt.imshow(grid_map, cmap=cmap, origin="upper")

    if start is not None:
        plt.scatter(start[0], start[1], color='green', s=100, marker='o', label='Start')
    if end is not None:
        plt.scatter(end[0], end[1], color='black', s=100, marker='*', label='End')

    if default_path:
        xs, ys = zip(*default_path)
        plt.plot(xs, ys, color='blue', linewidth=2, label='Default optimizer path (INSGA-style)')
        plt.scatter(xs, ys, color='blue', s=15)

    if user_path:
        xs, ys = zip(*user_path)
        plt.plot(xs, ys, color='red', linewidth=2, label='Preference-optimized path')
        plt.scatter(xs, ys, color='red', s=15)

    plt.legend(loc='upper right')
    plt.title("Final path visualization")
    plt.axis('off')
    path_out = save_path or os.path.join(figure_dir or FIGURE_DIR, "final_paths.png")
    plt.savefig(path_out, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(
    init_vals,
    best_vals,
    labels=None,
    save_path=None,
    figure_dir: Optional[str] = None,
):
    """Bar chart comparing initial vs best objective values."""
    if labels is None:
        labels = list(DEFAULT_RAW_OBJ_LABELS)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, init_vals, width, label='Initial')
    ax.bar(x + width / 2, best_vals, width, label='Best')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Objective value')
    ax.set_title('Initial vs Best path objective comparison')
    ax.legend()
    plt.tight_layout()
    path_out = save_path or os.path.join(figure_dir or FIGURE_DIR, "comparison.png")
    fig.savefig(path_out)
    plt.close(fig)

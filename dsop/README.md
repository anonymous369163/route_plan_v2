# dsop — Route Planning Sub-package

Multi-objective evolutionary route optimizer used by PILOT.

## Module Map

```
dsop/
  constants.py       Hyperparameters, terrain cost tables, lookup arrays
  perf_log.py        Wall-clock logging (_PerfSession / _PerfSpan)
  map_cache.py       Lazy-loaded terrain map, distance transforms, gradients
  preference.py      PreferenceGuidance dataclass, utility-model scoring
  path_metrics.py    Path encoding/decoding, multi-objective evaluation
  path_planners.py   A*, RRT, initial population construction
  path_operators.py  Crossover, mutation, VNS local-search operators
  population.py      NSGA-II environment selection, _run_insga4_core
  representative.py  Quality-anchored disagreement representative selection
  optimizer.py       Public API: run_optimizer / run_optimizer_continuous
  __init__.py        Re-exports every public symbol
```

Dependency order (no circular imports):

```
constants
  -> perf_log
  -> map_cache
  -> preference
  -> path_metrics
  -> path_planners
  -> path_operators
  -> population
  -> representative
  -> optimizer
```

## Public API

### Optimizer entry-points

```python
from dsop import run_optimizer, run_optimizer_continuous, visualize_solution_result
```

| Function | Purpose |
|---|---|
| `run_optimizer(...)` | Full run: A*/RRT init + NSGA-II + elite local search + optional plots |
| `run_optimizer_continuous(...)` | Stateless continuation (no plots, warm-start supported) |
| `visualize_solution_result(result, ...)` | Save terrain overview figure from a result dict |

**Common parameters**

| Parameter | Type | Description |
|---|---|---|
| `user_weights` | `list[float]` | Objective weights for selecting the display candidate |
| `start` / `end` | `(int, int)` | Grid coordinates; randomly sampled if `None` |
| `n_gen` | `int` | Number of NSGA-II generations |
| `guidance` | `PreferenceGuidance` | Optional utility model for preference guidance |
| `skip_plots` | `bool` | Skip figure output (default `True` in `run_optimizer`) |

**Result dict keys** (both functions)

| Key | Description |
|---|---|
| `pareto_X` / `pareto_F` | Non-dominated front chromosomes and objectives |
| `best_vars` / `best_path` | Best-selected chromosome and decoded waypoints |
| `best_metrics` | Raw objective vector of the best path |
| `pop_X` / `pop_F` | Full final population |
| `start` / `end` | Route endpoints used |
| `perf_log_path` | Path to the performance log file (if enabled) |

### Preference guidance

```python
from dsop import PreferenceGuidance

guidance = PreferenceGuidance(utility_model=my_model)
result = run_optimizer(..., guidance=guidance)
```

If `utility_model` is `None` the algorithm runs as pure NSGA-II with no
additional preference information.

### Representative path selection

```python
from dsop import select_representative_paths

X_list, F_list = select_representative_paths(
    pareto_F, pareto_X, weights=..., n_select=5,
    utility_model=guidance.utility_model,
)
```

### Reproducibility

```python
from dsop import set_reproducible_rng, REPRO_RANDOM_SEED, REPRO_FIXED_START, REPRO_FIXED_END

set_reproducible_rng(REPRO_RANDOM_SEED)
result = run_optimizer(start=REPRO_FIXED_START, end=REPRO_FIXED_END, n_gen=40)
```

## Running the full pipeline

```python
import dsop

dsop.set_reproducible_rng(42)
result = dsop.run_optimizer(
    user_weights=dsop.DEFAULT_WEIGHTS,
    start=dsop.REPRO_FIXED_START,
    end=dsop.REPRO_FIXED_END,
    n_gen=40,
    skip_plots=False,
)
print(result["best_metrics"])
```

## Performance logging

Set environment variable `TRAINING_DSOP_PERF_LOG=1` (default on) to write
per-run wall-clock logs to the `log/` directory.

Disable with `TRAINING_DSOP_PERF_LOG=0` or pass `perf_log=False` directly:

```python
result = dsop.run_optimizer(..., perf_log=False)
```

## Objective vector layout

| Index | Name | Direction | Description |
|---|---|---|---|
| 0 | `cost` | minimize | Total terrain + line-crossing cost |
| 1 | `length` | minimize | Total path length (grid cells) |
| 2 | `safety` | maximize | Safety clearance score (higher = safer) |
| 3 | `smoothness` | maximize | Path smoothness / bend penalty (higher = smoother) |

Objectives 2 and 3 are internally flipped before Pareto sorting so all four
are treated as minimisation problems.

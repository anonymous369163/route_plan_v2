"""Passive, preregistered SLFN parameter observers."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


BASELINE_SLFN_PARAMS: Dict[str, Any] = {
    "n_hidden": 16,
    "lambda_delta": 2.0,
    "lambda_cross": 0.05,
    "pair_prior_scale": 0.15,
}

LAMBDA_NL_BASE_SLFN_PARAMS: Dict[str, Any] = {
    "n_hidden": 32,
    "lambda_delta": 4.0,
    "lambda_cross": 0.1,
    "pair_prior_scale": 0.075,
    "lambda_nl": 0.1,
}

_REGISTERED_VARIANTS: tuple[tuple[str, str, Any], ...] = (
    ("n_hidden_8", "n_hidden", 8),
    ("n_hidden_32", "n_hidden", 32),
    ("lambda_delta_1", "lambda_delta", 1.0),
    ("lambda_delta_4", "lambda_delta", 4.0),
    ("lambda_cross_0p025", "lambda_cross", 0.025),
    ("lambda_cross_0p1", "lambda_cross", 0.10),
    ("pair_prior_scale_0p075", "pair_prior_scale", 0.075),
    ("pair_prior_scale_0p3", "pair_prior_scale", 0.30),
)

FACTORIAL_SLFN_VALUES: Dict[str, tuple[Any, ...]] = {
    "n_hidden": (8, 16, 32),
    "lambda_delta": (1.0, 2.0, 4.0),
    "lambda_cross": (0.025, 0.05, 0.10),
    "pair_prior_scale": (0.075, 0.15, 0.30),
}


def build_slfn_ofat_grid() -> Dict[str, Dict[str, Any]]:
    grid = {"baseline": dict(BASELINE_SLFN_PARAMS)}
    for name, field, value in _REGISTERED_VARIANTS:
        grid[name] = dict(BASELINE_SLFN_PARAMS, **{field: value})
    return grid


def _value_token(value: Any) -> str:
    return format(value, "g").replace(".", "p")


def build_slfn_factorial_grid() -> Dict[str, Dict[str, Any]]:
    """Return the deterministic 3^4 joint grid, with the paper default first."""
    grid: Dict[str, Dict[str, Any]] = {"baseline": dict(BASELINE_SLFN_PARAMS)}
    fields = tuple(FACTORIAL_SLFN_VALUES)
    for values in product(*(FACTORIAL_SLFN_VALUES[field] for field in fields)):
        config = dict(zip(fields, values))
        if config == BASELINE_SLFN_PARAMS:
            continue
        name = (
            f"h{_value_token(config['n_hidden'])}"
            f"_d{_value_token(config['lambda_delta'])}"
            f"_c{_value_token(config['lambda_cross'])}"
            f"_p{_value_token(config['pair_prior_scale'])}"
        )
        grid[name] = config
    return grid


def build_slfn_lambda_nl_grid() -> Dict[str, Dict[str, Any]]:
    """Return the frozen 1x/2x/4x nonlinear-residual regularization probe."""
    return {
        "baseline": dict(LAMBDA_NL_BASE_SLFN_PARAMS),
        "lambda_nl_2x": dict(LAMBDA_NL_BASE_SLFN_PARAMS, lambda_nl=0.2),
        "lambda_nl_4x": dict(LAMBDA_NL_BASE_SLFN_PARAMS, lambda_nl=0.4),
    }


def validate_slfn_parameter_grid(
    configs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Validate any registered OFAT, factorial, or lambda_nl subset."""
    if not isinstance(configs, Mapping) or not configs:
        raise ValueError("SLFN parameter grid must be a non-empty mapping")
    uses_lambda_nl = any(
        isinstance(config, Mapping) and "lambda_nl" in config
        for config in configs.values()
    )
    registered = (
        build_slfn_lambda_nl_grid()
        if uses_lambda_nl
        else {**build_slfn_factorial_grid(), **build_slfn_ofat_grid()}
    )
    expected_fields = set(registered["baseline"])
    validated: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_config in configs.items():
        name = str(raw_name)
        if name not in registered:
            raise ValueError(f"{name!r} is not in a registered SLFN parameter grid")
        if not isinstance(raw_config, Mapping) or set(raw_config) != expected_fields:
            raise ValueError(
                f"SLFN config {name!r} must contain exactly {sorted(expected_fields)}"
            )
        config = {
            "n_hidden": int(raw_config["n_hidden"]),
            "lambda_delta": float(raw_config["lambda_delta"]),
            "lambda_cross": float(raw_config["lambda_cross"]),
            "pair_prior_scale": float(raw_config["pair_prior_scale"]),
        }
        if uses_lambda_nl:
            config["lambda_nl"] = float(raw_config["lambda_nl"])
        if config["n_hidden"] <= 0 or not np.all(
            np.isfinite(
                [
                    config["lambda_delta"],
                    config["lambda_cross"],
                    config["pair_prior_scale"],
                    *([config["lambda_nl"]] if uses_lambda_nl else []),
                ]
            )
        ) or (uses_lambda_nl and config["lambda_nl"] < 0.0):
            raise ValueError(f"SLFN config {name!r} contains invalid values")
        if config != registered[name]:
            raise ValueError(f"SLFN config {name!r} differs from its registered values")
        validated[name] = config
    if "baseline" not in validated or validated["baseline"] != registered["baseline"]:
        raise ValueError("SLFN parameter grid must include the registered baseline")
    return validated


def validate_slfn_ofat_grid(
    configs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(configs, Mapping) or not configs:
        raise ValueError("SLFN OFAT grid must be a non-empty mapping")
    registered = build_slfn_ofat_grid()
    validated: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_config in configs.items():
        name = str(raw_name)
        if name not in registered:
            raise ValueError(f"{name!r} is not in the registered SLFN OFAT grid")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"SLFN OFAT config {name!r} must be a mapping")
        if set(raw_config) != set(BASELINE_SLFN_PARAMS):
            raise ValueError(
                f"SLFN OFAT config {name!r} must contain exactly "
                f"{sorted(BASELINE_SLFN_PARAMS)}"
            )
        config = {
            "n_hidden": int(raw_config["n_hidden"]),
            "lambda_delta": float(raw_config["lambda_delta"]),
            "lambda_cross": float(raw_config["lambda_cross"]),
            "pair_prior_scale": float(raw_config["pair_prior_scale"]),
        }
        if config["n_hidden"] <= 0 or not np.all(
            np.isfinite(
                [
                    config["lambda_delta"],
                    config["lambda_cross"],
                    config["pair_prior_scale"],
                ]
            )
        ):
            raise ValueError(f"SLFN OFAT config {name!r} contains invalid values")
        if any(config[key] != registered[name][key] for key in BASELINE_SLFN_PARAMS):
            raise ValueError(f"SLFN OFAT config {name!r} differs from the registered grid")
        differences = sum(
            config[key] != BASELINE_SLFN_PARAMS[key]
            for key in BASELINE_SLFN_PARAMS
        )
        expected_differences = 0 if name == "baseline" else 1
        if differences != expected_differences:
            raise ValueError(
                f"SLFN OFAT config {name!r} violates the one-factor-at-a-time rule"
            )
        validated[name] = config
    if "baseline" not in validated:
        raise ValueError("SLFN OFAT grid must include the registered baseline")
    return validated


class SLFNParameterShadowBank:
    """Train registered SLFN variants on one event stream without producing decisions."""

    def __init__(
        self,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        shared_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        from preference_utility import build_pairwise_utility_learner

        self.configs = validate_slfn_parameter_grid(configs)
        common = dict(shared_kwargs or {})
        common.pop("rng_seed", None)
        config_fields = {field for config in self.configs.values() for field in config}
        for field in config_fields:
            common.pop(field, None)
        self.learners = {
            name: build_pairwise_utility_learner(
                variant="benchmark_slfn",
                **dict(common, **config, rng_seed=42),
            )
            for name, config in self.configs.items()
        }

    def set_reference(self, rows: Sequence[Sequence[float]]) -> None:
        for learner in self.learners.values():
            learner.set_reference(rows)

    def set_reference_from_bounds(
        self,
        ref_min: Sequence[float],
        ref_max: Sequence[float],
    ) -> None:
        for learner in self.learners.values():
            learner.set_reference_from_bounds(ref_min, ref_max)

    def set_prior(self, weights: Sequence[float], confidence: float = 1.0) -> None:
        for learner in self.learners.values():
            learner.set_prior(weights, confidence=confidence)

    def set_structure_prior(
        self,
        pair_priors: Optional[Sequence[float]] = None,
        confidence: float = 0.0,
        pair_prior_scale: Optional[float] = None,
    ) -> None:
        for learner in self.learners.values():
            learner.set_structure_prior(
                pair_priors,
                confidence=confidence,
                pair_prior_scale=pair_prior_scale,
            )

    def set_importance_order(
        self,
        order_pairs: Sequence[Tuple[int, int]],
        confidence: float = 0.5,
    ) -> None:
        for learner in self.learners.values():
            learner.set_importance_order(order_pairs, confidence=confidence)

    def update(
        self,
        x_list: Sequence[Sequence[Sequence[float]]],
        y_list: Sequence[int],
        round_idx: int = 0,
        total_rounds: int = 3,
    ) -> None:
        for learner in self.learners.values():
            learner.update(
                x_list,
                y_list,
                round_idx=round_idx,
                total_rounds=total_rounds,
            )

    def snapshot_archive(
        self,
        archive_f_raw_all_min: Sequence[Sequence[float]],
        archive_paths: Sequence[Sequence[Sequence[float]]],
        *,
        shortlist_size: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        archive_f = np.asarray(archive_f_raw_all_min, dtype=float)
        if archive_f.ndim == 1:
            archive_f = archive_f[None, :]
        paths = list(archive_paths)
        if len(archive_f) == 0 or len(paths) != len(archive_f):
            raise RuntimeError("SLFN parameter shadow requires an aligned non-empty archive")
        k = min(max(1, int(shortlist_size)), len(archive_f))
        snapshots: Dict[str, Dict[str, Any]] = {}
        for name, learner in self.learners.items():
            scores = np.asarray(learner.score_raw_batch(archive_f), dtype=float).reshape(-1)
            if len(scores) != len(archive_f) or not np.all(np.isfinite(scores)):
                raise RuntimeError(
                    f"SLFN parameter shadow {name!r} produced invalid archive scores"
                )
            indices = np.argsort(scores, kind="stable")[:k].astype(int).tolist()
            snapshots[name] = {
                "config": dict(self.configs[name]),
                "final_utility_params": learner.get_params(),
                "archive_learned_scores": [float(value) for value in scores],
                "shortlist_archive_indices": indices,
                "shortlist_F_raw_all_min": archive_f[
                    np.asarray(indices, dtype=int)
                ].tolist(),
                "shortlist_paths": [
                    [tuple(point) for point in paths[index]] for index in indices
                ],
            }
        return snapshots

    def get_params(self) -> Dict[str, Any]:
        return {
            "variant": "benchmark_slfn_parameter_shadow_bank",
            "configs": {name: dict(config) for name, config in self.configs.items()},
            "members": {
                name: learner.get_params() for name, learner in self.learners.items()
            },
        }

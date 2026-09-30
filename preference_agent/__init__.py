"""Public exports for the simplified preference-agent mainline."""

from .constants import DEFAULT_WEIGHTS, DIRECTIONS, OBJECTIVES
from .main import run_full_session
from .models import RoundTrace, SessionConfig, SessionResult
from .scoring import canonicalize_best_metrics, canonicalize_pareto_metrics_batch
from .utils import normalize_weights

__all__ = [
    "DEFAULT_WEIGHTS",
    "DIRECTIONS",
    "OBJECTIVES",
    "RoundTrace",
    "SessionConfig",
    "SessionResult",
    "canonicalize_best_metrics",
    "canonicalize_pareto_metrics_batch",
    "normalize_weights",
    "run_full_session",
]

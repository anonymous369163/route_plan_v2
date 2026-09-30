"""Run PILOT on one transmission-line routing case and plot the recommended route.

The decision maker is simulated by a hidden Tchebycheff utility with weights
``TRUE_WEIGHTS``; PILOT only sees its choices from the query menus. With
``OPENAI_API_KEY`` set, an LLM turns the choice history into a structural
preference prior; without it the SLFN utility is learned from choices alone.

    python demo.py                        # paper budget: 12 rounds, 10 generations per round
    python demo.py --gens-per-round 2     # quick check with a smaller search budget
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_defaults import FORMAL_DEFAULT_LLM_MODEL, build_proposed_run_full_session_kwargs
from preference_agent import canonicalize_pareto_metrics_batch, run_full_session
from preference_agent.bayesian_top1 import calibrate_session_top1
from preference_agent.constants import OBJECTIVES
from preference_agent.llm_interface import maybe_get_client
from preference_agent.scoring import tchebycheff_score_clipped
from scenario_visualization import draw_paths_on_map_axes, load_routing_grid_map

# Showcase case "Rand_30" of the paper (map cells are (x, y)).
START, END = (7, 234), (445, 169)
TRUE_WEIGHTS = [0.19755671613855805, 0.6739649931063388, 0.012483700833670393, 0.11599458992143269]
SEED = 63383

# Final PILOT configuration used for all reported TLR results.
FINAL_CONFIG = {
    "query_selector_mode": "simple",
    "utility_ensemble_mode": "shadow_final",
    "shadow_prior_member_count": 5,
    "learning_query_allocation": "terminal_11_plus_1_geometric_top5",
    "temporal_weight_ratio": 5.0,
    "noise_kappa": -1.0,
}


def run_pilot(start, end, true_weights, seed, *, llm_model, output_dir, gens_per_round=None):
    np.random.seed(seed)
    random.seed(seed)
    kwargs = build_proposed_run_full_session_kwargs(llm_model=llm_model, output_dir=output_dir)
    kwargs.update(FINAL_CONFIG)
    if gens_per_round is not None:
        kwargs["n_gen_warmup"] = kwargs["n_gen_per_interaction"] = int(gens_per_round)
    result = run_full_session(true_weights=true_weights, start=start, end=end, **kwargs)
    top1 = calibrate_session_top1(result)
    return result, top1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gens-per-round", type=int, default=None, help="generations between consultations (paper: 10)")
    parser.add_argument("--llm-model", default=FORMAL_DEFAULT_LLM_MODEL)
    parser.add_argument("--output-dir", default="demo_output")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if maybe_get_client() is None:
        print("OPENAI_API_KEY not set: running without the LLM structural prior.")
    result, top1 = run_pilot(START, END, TRUE_WEIGHTS, SEED, llm_model=args.llm_model,
                             output_dir=str(out / "session"), gens_per_round=args.gens_per_round)

    archive = canonicalize_pareto_metrics_batch(np.asarray(result.final_archive_metrics_raw, dtype=float))
    mins, maxs = archive.min(axis=0), archive.max(axis=0)
    psi_all = tchebycheff_score_clipped(archive, TRUE_WEIGHTS, mins, maxs)
    idx = int(top1["selected_index"])
    summary = {
        "recommended_route": dict(zip(OBJECTIVES, map(float, archive[idx]))),
        "psi_recommended": float(psi_all[idx]),
        "psi_best_in_archive": float(np.min(psi_all)),
        "archive_size": int(len(archive)),
        "llm_valid_rounds": len(result.structure_valid_rounds or []),
        "rounds": len(result.round_traces or []),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_paths_on_map_axes(ax, load_routing_grid_map(), START, END,
                           [{"path": result.final_archive_paths[idx], "color": "#dc2626", "label": "PILOT recommendation"}])
    ax.set_title(f"PILOT recommended route (Psi = {summary['psi_recommended']:.3f}, lower is better)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "route.png", dpi=150)
    print(f"Wrote {out / 'route.png'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()

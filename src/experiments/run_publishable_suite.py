"""
Run the publishable research suite end-to-end:
1) Leakage-safe protocol evaluation (random + time, optional cross-dataset)
2) Publication bundle generation (tables + claim checks + reproducibility manifest)
"""

from __future__ import annotations

import argparse

from src.experiments.build_publication_bundle import build_publication_bundle
from src.experiments.run_research_protocols import run_research_protocols


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run publishable NetCausalAI research suite.")
    parser.add_argument("--include-cross-dataset", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--split-min-test-anomalies", type=int, default=100)
    parser.add_argument("--split-min-val-anomalies", type=int, default=100)
    parser.add_argument("--threshold-objective", type=str, default="f0.5")
    parser.add_argument("--score-model-preference", type=str, default="temporal_only")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_research_protocols(
        include_cross_dataset=bool(args.include_cross_dataset),
        n_bootstrap=int(args.n_bootstrap),
        split_min_test_anomalies=int(args.split_min_test_anomalies),
        split_min_val_anomalies=int(args.split_min_val_anomalies),
        threshold_objective=str(args.threshold_objective),
        score_model_preference=str(args.score_model_preference),
    )
    build_publication_bundle()

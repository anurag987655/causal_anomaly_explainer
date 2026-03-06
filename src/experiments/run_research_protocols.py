"""
Research protocol runner for NetCausalAI.

Protocols:
1) Random split (session-level stratified)
2) Time split (train/validation on earlier sessions, test on later sessions)
3) Cross-dataset benchmark (optional, heavier run)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.casual.evaluate_results import evaluate_performance
from src.experiments.run_cross_dataset_experiments import run_all_experiments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = PROJECT_ROOT / "data" / "experiments" / "research_protocols"


def _run_single_protocol(
    name: str,
    split_strategy: str,
    n_bootstrap: int,
    split_min_test_anomalies: int,
    split_min_val_anomalies: int,
    threshold_objective: str,
    score_model_preference: str,
) -> Dict:
    output_dir = PROTOCOL_ROOT / name
    metrics = evaluate_performance(
        ground_truth_path=str(PROJECT_ROOT / "data" / "processed" / "all_sessions_detailed.csv"),
        predictions_path=str(PROJECT_ROOT / "data" / "processed" / "anomaly_scores_with_features.csv"),
        output_dir=str(output_dir),
        threshold_objective=str(threshold_objective),
        learn_hybrid_weights=(str(score_model_preference).strip().lower() in {"auto", "learned_hybrid"}),
        learn_family_thresholds=False,  # leakage-safe headline protocol
        split_strategy=split_strategy,
        n_bootstrap=n_bootstrap,
        split_min_test_anomalies=split_min_test_anomalies,
        split_min_val_anomalies=split_min_val_anomalies,
        score_model_preference=str(score_model_preference),
    )
    row = {
        "protocol": name,
        "split_strategy": split_strategy,
        "success": bool(metrics and metrics.get("success")),
    }
    if metrics and metrics.get("success"):
        row.update(
            {
                "effective_split_strategy": metrics.get("split_strategy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "auc_roc": metrics.get("auc_roc"),
                "pr_auc": metrics.get("pr_auc"),
                "threshold": metrics.get("threshold"),
                "score_model": metrics.get("score_model"),
                "threshold_objective": metrics.get("threshold_objective"),
                "report_path": metrics.get("report_path"),
            }
        )
    else:
        row["error"] = metrics.get("error") if isinstance(metrics, dict) else "Unknown error"
    return row


def run_research_protocols(
    include_cross_dataset: bool = False,
    n_bootstrap: int = 200,
    split_min_test_anomalies: int = 100,
    split_min_val_anomalies: int = 100,
    threshold_objective: str = "f0.5",
    score_model_preference: str = "temporal_only",
) -> pd.DataFrame:
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    rows.append(
        _run_single_protocol(
            name="random_split",
            split_strategy="random",
            n_bootstrap=n_bootstrap,
            split_min_test_anomalies=split_min_test_anomalies,
            split_min_val_anomalies=split_min_val_anomalies,
            threshold_objective=threshold_objective,
            score_model_preference=score_model_preference,
        )
    )
    rows.append(
        _run_single_protocol(
            name="time_split",
            split_strategy="time",
            n_bootstrap=n_bootstrap,
            split_min_test_anomalies=split_min_test_anomalies,
            split_min_val_anomalies=split_min_val_anomalies,
            threshold_objective=threshold_objective,
            score_model_preference=score_model_preference,
        )
    )

    if include_cross_dataset:
        summary = run_all_experiments(
            threshold_objective=str(threshold_objective),
            score_model_preference=str(score_model_preference),
            n_bootstrap=int(n_bootstrap),
        )
        rows.append(
            {
                "protocol": "cross_dataset",
                "split_strategy": "cross_dataset",
                "success": not summary.empty,
                "macro_f1": float(summary["f1"].dropna().mean()) if "f1" in summary.columns else None,
                "macro_auc_roc": float(summary["auc_roc"].dropna().mean()) if "auc_roc" in summary.columns else None,
                "summary_csv": str(PROJECT_ROOT / "data" / "experiments" / "cross_dataset_summary.csv"),
            }
        )

    out_df = pd.DataFrame(rows)
    summary_csv = PROTOCOL_ROOT / "protocol_summary.csv"
    summary_md = PROTOCOL_ROOT / "protocol_summary.md"
    out_df.to_csv(summary_csv, index=False)
    try:
        out_df.to_markdown(summary_md, index=False)
    except Exception:
        with open(summary_md, "w") as f:
            f.write("# Research protocol summary\n\n")
            f.write(out_df.to_csv(index=False))

    print("\n=== Research protocol summary ===")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {summary_md}")
    return out_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe research protocols.")
    parser.add_argument(
        "--include-cross-dataset",
        action="store_true",
        help="Run the cross-dataset benchmark in addition to random/time split evaluations.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Bootstrap rounds for CI in each protocol evaluation.",
    )
    parser.add_argument(
        "--split-min-test-anomalies",
        type=int,
        default=100,
        help="Minimum anomaly sessions required in the test window for temporal split.",
    )
    parser.add_argument(
        "--split-min-val-anomalies",
        type=int,
        default=100,
        help="Minimum anomaly sessions required in the validation window for temporal split.",
    )
    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="f0.5",
        help="Validation threshold objective. Example: f0.5 (precision-weighted), f1, f2.",
    )
    parser.add_argument(
        "--score-model-preference",
        type=str,
        default="temporal_only",
        help="Score model to deploy: temporal_only, full_hybrid, learned_hybrid, markov_only, statistical_only, hybrid_without_stat, or auto.",
    )
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

"""
Run cross-dataset NetCausalAI experiments on multiple CICIDS CSV files.

Pipeline per experiment:
1) Build flow-level evaluation sessions from a target dataset
2) Score with trained Markov model
3) Evaluate detection metrics (validation-tuned threshold, held-out test)
4) Export consolidated comparison report
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.ingestion.prepare_baseline import prepare_baseline
from src.ingestion.flow_event_mapper import infer_flow_columns, extract_flow_features, build_session_rows
from src.casual.build_dag import main as build_main
from src.casual.anamoly_scoring import run_anomaly_scoring_with_rca
from src.casual.evaluate_results import evaluate_performance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "experiments"

# Datasets chosen because they contain meaningful attack labels.
EXPERIMENT_DATASETS: List[Tuple[str, str]] = [
    ("tuesday_patator", "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("wednesday_dos", "Wednesday-workingHours.pcap_ISCX.csv"),
    ("thursday_webattack", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("friday_morning_bot", "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("friday_afternoon_portscan", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("friday_afternoon_ddos", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    ("thursday_infiltration", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
]


def _pick_col(df: pd.DataFrame, candidates: List[str], default: str | None = None) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return default


def _sample_balanced_by_label(
    csv_path: Path,
    max_benign: int = 25000,
    max_attack: int = 25000,
    chunksize: int = 100000,
) -> pd.DataFrame:
    benign_parts: List[pd.DataFrame] = []
    attack_parts: List[pd.DataFrame] = []
    benign_left = max_benign
    attack_left = max_attack

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        label_col = _pick_col(chunk, ["Label", "label"], None)
        if label_col is None:
            continue

        labels = chunk[label_col].astype(str).str.strip().str.upper()
        benign_chunk = chunk.loc[labels == "BENIGN"]
        attack_chunk = chunk.loc[labels != "BENIGN"]

        if benign_left > 0 and not benign_chunk.empty:
            take = benign_chunk.head(benign_left)
            benign_parts.append(take)
            benign_left -= len(take)

        if attack_left > 0 and not attack_chunk.empty:
            take = attack_chunk.head(attack_left)
            attack_parts.append(take)
            attack_left -= len(take)

        if benign_left <= 0 and attack_left <= 0:
            break

    selected = []
    if benign_parts:
        selected.append(pd.concat(benign_parts, ignore_index=True))
    if attack_parts:
        selected.append(pd.concat(attack_parts, ignore_index=True))
    if not selected:
        raise ValueError(f"No labeled rows found in {csv_path}")

    df = pd.concat(selected, ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _flow_rows_to_event_sessions(df: pd.DataFrame, dataset_tag: str) -> pd.DataFrame:
    colmap = infer_flow_columns(df)

    events = []
    for idx, row in df.iterrows():
        flow = extract_flow_features(row, colmap, fallback_label="BENIGN")
        session_id = (
            f"{dataset_tag}_{flow['src_ip']}_{flow['dst_ip']}_"
            f"{flow['src_port']}_{flow['dst_port']}_{flow['protocol']}_{idx}"
        )
        events.extend(build_session_rows(session_id, flow))

    return pd.DataFrame(events)


def _build_eval_sessions_for_file(csv_name: str, output_csv: Path, dataset_tag: str) -> Dict:
    csv_path = RAW_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    sampled = _sample_balanced_by_label(csv_path)
    sessions = _flow_rows_to_event_sessions(sampled, dataset_tag=dataset_tag)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_csv, index=False)

    session_labels = sessions.groupby("session_id")["label"].first().astype(str).str.strip()
    label_counts = session_labels.value_counts().to_dict()
    return {
        "rows": int(len(sessions)),
        "sessions": int(sessions["session_id"].nunique()),
        "label_counts": label_counts,
    }


def run_all_experiments(
    threshold_objective: str = "f0.5",
    score_model_preference: str = "temporal_only",
    n_bootstrap: int = 100,
) -> pd.DataFrame:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    # Train baseline model once.
    print("=== Preparing baseline and training model ===")
    prepare_baseline()
    build_main(train_csv="data/processed/baseline_training.csv")

    model_path = "data/processed/dag_model_complete.pkl"
    rows = []

    for exp_name, csv_name in EXPERIMENT_DATASETS:
        print(f"\n=== Experiment: {exp_name} ({csv_name}) ===")
        exp_dir = EXPERIMENT_ROOT / exp_name
        sessions_csv = exp_dir / "all_sessions_detailed.csv"
        scores_csv = exp_dir / "anomaly_scores_with_features.csv"
        results_dir = exp_dir / "results"

        dataset_stats = _build_eval_sessions_for_file(csv_name, sessions_csv, dataset_tag=exp_name)

        run_anomaly_scoring_with_rca(
            sessions_csv=str(sessions_csv),
            dag_model=model_path,
            output_csv=str(scores_csv),
        )

        metrics = evaluate_performance(
            ground_truth_path=str(sessions_csv),
            predictions_path=str(scores_csv),
            output_dir=str(results_dir),
            threshold_objective=str(threshold_objective),
            learn_hybrid_weights=(str(score_model_preference).strip().lower() in {"auto", "learned_hybrid"}),
            learn_family_thresholds=False,
            split_strategy="random",
            n_bootstrap=int(n_bootstrap),
            score_model_preference=str(score_model_preference),
        )

        row = {
            "experiment": exp_name,
            "source_file": csv_name,
            "sessions": dataset_stats["sessions"],
            "events": dataset_stats["rows"],
            "label_counts": str(dataset_stats["label_counts"]),
        }
        if metrics and metrics.get("success"):
            row.update({
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc_roc": metrics["auc_roc"],
                "pr_auc": metrics.get("pr_auc"),
                "calibrated_pr_auc": metrics.get("calibrated_pr_auc"),
                "brier_score": metrics.get("brier_score"),
                "ece": metrics.get("ece"),
                "threshold": metrics["threshold"],
                "threshold_objective": metrics.get("threshold_objective"),
                "score_model": metrics.get("score_model"),
                "score_orientation": metrics["score_orientation"],
                "evaluation_sessions": metrics["evaluation_sessions"],
                "evaluation_anomalies": metrics["evaluation_anomalies"],
            })
        else:
            row.update({
                "precision": None,
                "recall": None,
                "f1": None,
                "auc_roc": None,
                "pr_auc": None,
                "calibrated_pr_auc": None,
                "brier_score": None,
                "ece": None,
                "threshold": None,
                "threshold_objective": None,
                "score_model": None,
                "score_orientation": None,
                "evaluation_sessions": None,
                "evaluation_anomalies": None,
                "error": metrics.get("error") if isinstance(metrics, dict) else "Unknown error",
            })

        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(by="auc_roc", ascending=False, na_position="last")
    summary_csv = EXPERIMENT_ROOT / "cross_dataset_summary.csv"
    summary_md = EXPERIMENT_ROOT / "cross_dataset_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    try:
        summary_df.to_markdown(summary_md, index=False)
    except Exception:
        # Fallback when optional dependency "tabulate" is not installed.
        with open(summary_md, "w") as f:
            f.write("# Cross-dataset summary\n\n")
            f.write(summary_df.to_csv(index=False))

    print("\n=== Cross-dataset summary ===")
    print(summary_df[["experiment", "precision", "recall", "f1", "auc_roc", "pr_auc", "threshold"]].to_string(index=False))
    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {summary_md}")
    return summary_df


if __name__ == "__main__":
    run_all_experiments()

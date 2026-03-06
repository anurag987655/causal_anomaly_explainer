"""
Complete NetCausalAI Pipeline:
1. Parse PCAPs → Detailed sessions
2. Build DAG + Features → Session features
3. Anomaly Scoring → 0-100 scores
4. Research Evaluation → Metrics + CI + Ablations
5. Root Cause Analysis → Why each anomaly
6. Behavioral Clustering → Behavior families
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def run_complete_analysis():
    """Execute full pipeline: Detection → RCA → Clustering"""
    
    print("\n" + "="*60)
    print("NETCAUSALAI - RESEARCH PERFORMANCE MODE")
    print("="*60 + "\n")
    
    # Step 0: Prepare Baseline and Mixed Data
    print("[STEP 0/5] Preparing Research Datasets...")
    from src.ingestion.prepare_baseline import prepare_baseline
    from src.ingestion.mixed_data import load_and_transform_combined
    
    prepare_baseline()
    load_and_transform_combined()
    
    # Step 1: Build DAG using ONLY Baseline Data
    print("\n[STEP 1/5] Training Causal Model on Pure Baseline...")
    from src.casual.build_dag import main as build_main
    build_main(train_csv="data/processed/baseline_training.csv")
    
    # Step 2: Calculate Anomaly Scores on Mixed Data
    print("\n[STEP 2/5] Scoring Multi-Attack Dataset...")
    from src.casual.anamoly_scoring import run_anomaly_scoring_with_rca
    run_anomaly_scoring_with_rca(
        sessions_csv="data/processed/all_sessions_detailed.csv",
        train_features="data/processed/baseline_features.csv",
        dag_model="data/processed/dag_model_complete.pkl",
        generate_rca=False
    )
    
    # Step 3: Research Evaluation
    print("\n[STEP 3/6] Running research-grade evaluation...")
    from src.casual.evaluate_results import evaluate_performance
    eval_metrics = evaluate_performance(
        ground_truth_path="data/processed/all_sessions_detailed.csv",
        predictions_path="data/processed/anomaly_scores_with_features.csv",
        output_dir="data/results",
        threshold_objective="f1",
        learn_hybrid_weights=True,
        learn_family_thresholds=False,
        split_strategy="random",
    )

    policy_path = None
    if isinstance(eval_metrics, dict):
        policy_path = eval_metrics.get("deployment_policy_path")
    if policy_path and os.path.exists(policy_path):
        print("\n[STEP 3.5/6] Re-scoring with deployment policy artifact...")
        run_anomaly_scoring_with_rca(
            sessions_csv="data/processed/all_sessions_detailed.csv",
            train_features="data/processed/baseline_features.csv",
            dag_model="data/processed/dag_model_complete.pkl",
            output_csv="data/processed/anomaly_scores_with_features.csv",
            deployment_policy=policy_path,
            generate_rca=False,
        )

    # Step 4: Root Cause Analysis
    print("\n[STEP 4/6] Generating RCA explanations...")
    from src.casual.root_cause_analysis import run_rca_pipeline
    rca_results = run_rca_pipeline(
        sessions_csv="data/processed/all_sessions_detailed.csv",
        scores_csv="data/processed/anomaly_scores_with_features.csv",
        dag_model="data/processed/dag_model_complete.pkl",
        limit=200  # Explain top 200 anomalies
    )

    # Step 5: Behavioral Clustering
    print("\n[STEP 5/6] Clustering behavior families...")
    from src.casual.behavioral_clustering import run_clustering_pipeline
    clustering_results = run_clustering_pipeline(
        rca_csv="data/processed/rca_explanations.csv",
        output_dir="data/processed",
        min_cluster_size=5,  # Campaign threshold
        visualize=True
    )
    
    print("\n" + "="*60)
    print("✅ COMPLETE PIPELINE FINISHED")
    print("="*60)
    print("\n📁 Output Files:")
    print("  • data/processed/all_sessions_detailed.csv - Raw session data")
    print("  • data/processed/session_features_for_clustering.csv - 25+ features per session")
    print("  • data/processed/dag_model_complete.pkl - Full Markov model")
    print("  • data/processed/anomaly_scores_with_features.csv - 0-100 scores")
    print("  • data/results/evaluation_report.txt - Research-grade metrics")
    print("  • data/results/ablation_results.csv - Model ablations")
    print("  • data/results/per_label_analysis.csv - Attack-family behavior")
    print("  • data/results/per_family_error_analysis.csv - Family-level misses/errors")
    print("  • data/results/learned_hybrid_weights.csv - Validation-learned score weights")
    print("  • data/processed/rca_explanations.csv - Root cause analysis")
    print("  • data/processed/rca_summary.txt - Human-readable RCA summary")
    print("  • data/processed/clustering_results.csv - Sessions with cluster IDs")
    print("  • data/processed/cluster_interpretations.json - Behavior family profiles")
    print("  • data/processed/clustering_summary.txt - Human-readable cluster summary")
    print("  • data/processed/cluster_visualization.png - 2D cluster visualization")
    print("  • data/processed/clusterer_model.pkl - Trained HDBSCAN model")
    print("  • data/processed/active_campaigns.csv - Ongoing campaigns (>5 sessions)")

if __name__ == "__main__":
    run_complete_analysis()

# NetCausalAI Update Log

## Day 1 - Pipeline Trace and Data Flow Documentation

Date: 2026-02-16
Scope: Traced `src/run_full_pipeline.py` execution order and documented each module's inputs/outputs and data split points.

### Execution order (from `src/run_full_pipeline.py`)
1. `src/ingestion/prepare_baseline.py` (`prepare_baseline`)
2. `src/ingestion/mixed_data.py` (`load_and_transform_combined`)
3. `src/casual/build_dag.py` (`main(train_csv="data/processed/baseline_training.csv")`)
4. `src/casual/anamoly_scoring.py` (`run_anomaly_scoring_with_rca`)
5. `src/casual/root_cause_analysis.py` (`run_rca_pipeline`)
6. `src/casual/behavioral_clustering.py` (`run_clustering_pipeline`)

### Module I/O mapping

1. `prepare_baseline.py`
- Input: `data/raw/Monday-WorkingHours.pcap_ISCX.csv` (first 100000 rows via `.head(100000)`).
- Processing: infers flow columns, extracts flow features, builds per-flow session event rows.
- Output: `data/processed/baseline_training.csv`.
- Purpose: creates baseline-only sessions used to train DAG and derive baseline features.

2. `mixed_data.py`
- Input files under `data/raw`:
  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (label hint `WebAttack`, first 50000 rows)
  - `Tuesday-WorkingHours.pcap_ISCX.csv` (label hint `BENIGN`, first 50000 rows)
- Processing: same flow-to-session event construction, combining both datasets.
- Output: `data/processed/all_sessions_detailed.csv`.
- Purpose: creates mixed benign/attack sessions for scoring, RCA, and clustering.

3. `build_dag.py`
- Input in this pipeline run: `data/processed/baseline_training.csv` (passed as `train_csv`).
- Processing:
  - counts event transitions per session
  - computes transition probabilities/confidence
  - builds DAG graph with edge/node attributes
  - computes per-session feature vectors
- Outputs:
  - `data/processed/dag_model_complete.pkl`
  - `data/processed/dag_edges_complete.csv`
  - `data/processed/session_features_for_clustering.csv`
  - `data/processed/feature_summary.json`
  - `data/processed/dag_graph_complete.png`
  - `data/processed/baseline_features.csv` (saved because `train_csv` is provided)
- Purpose: trains causal transition model on baseline sessions and emits baseline feature set.

4. `anamoly_scoring.py`
- Inputs:
  - sessions: `data/processed/all_sessions_detailed.csv`
  - DAG model: `data/processed/dag_model_complete.pkl`
  - training features: `data/processed/baseline_features.csv`
- Processing:
  - computes structural/intensity/regularity/statistical components
  - combines into `anomaly_score_0_100`
  - flags anomalies by percentile threshold
- Output: `data/processed/anomaly_scores_with_features.csv`
- Purpose: scores mixed sessions using baseline-trained model + baseline-trained IsolationForest.

5. `root_cause_analysis.py`
- Inputs:
  - sessions: `data/processed/all_sessions_detailed.csv`
  - scores: `data/processed/anomaly_scores_with_features.csv`
  - model: `data/processed/dag_model_complete.pkl`
- Processing:
  - selects anomalous sessions (`is_anomaly == True`)
  - explains top-N anomalies (`limit=200` from pipeline call)
  - derives drivers, narratives, and cluster hints
- Outputs:
  - `data/processed/rca_explanations.csv`
  - `data/processed/rca_explanations.json`
  - `data/processed/rca_summary.txt`
  - `data/processed/rca_cluster_hints.csv`
- Purpose: converts anomaly scores into root-cause explanations.

6. `behavioral_clustering.py`
- Input: `data/processed/rca_explanations.csv`
- Processing:
  - selects RCA numeric features
  - robust scales
  - HDBSCAN clustering
  - cluster interpretation + optional visualization
- Outputs:
  - `data/processed/clustering_results.csv`
  - `data/processed/cluster_interpretations.json`
  - `data/processed/clustering_summary.txt`
  - `data/processed/clusterer_model.pkl`
  - optional `data/processed/active_campaigns.csv` (if campaigns exist)
  - `data/processed/cluster_visualization.png` (visualize=True)
- Purpose: groups anomalous sessions into behavior families/campaigns.

### Where data splits happen (train/validation/test)
- Explicit train split:
  - Baseline training split is separate by source file and used only for model fitting:
    - `prepare_baseline.py` -> `baseline_training.csv`
    - `build_dag.py` trains on `baseline_training.csv`
    - `anamoly_scoring.py` fits IsolationForest on `baseline_features.csv` (derived from baseline training set)
- Inference/evaluation split:
  - Mixed dataset (`all_sessions_detailed.csv`) is scored and analyzed, distinct from baseline training set.
- Row caps (sampling-style subsetting):
  - `prepare_baseline.py`: `.head(100000)`
  - `mixed_data.py`: `.head(50000)` per source file
- Anomaly selection split:
  - `anamoly_scoring.py` marks top percentile (`anomaly_percentile`, default 5%) using quantile threshold.
- RCA selection split:
  - `root_cause_analysis.py` processes only rows where `is_anomaly` is true, then applies `limit` (200 in full pipeline call).
- Validation/test note:
  - No explicit validation/test split logic exists inside this full pipeline path (`run_full_pipeline.py` and called modules).

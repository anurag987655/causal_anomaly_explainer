# NetCausalAI

NetCausalAI is a network anomaly detection and behavioral analysis pipeline for research on intrusion detection.  
It combines:
- sequence-based behavior modeling (Markov transition model),
- hybrid anomaly scoring (Markov + Isolation Forest),
- root-cause explanation (RCA),
- and behavior-family clustering.

The project is designed to run on CICIDS-style network flow CSVs and produce reproducible experiment reports.

## What Problem This Solves

Traditional anomaly scores alone are hard to operationalize in security workflows.  
NetCausalAI aims to provide:
- anomaly ranking (`0-100` score),
- why a session is anomalous (RCA drivers),
- and grouping of related anomalous sessions into behavior families.

This makes it easier to build research results that are both quantitative (AUC/F1) and explainable.

## High-Level Pipeline

1. Ingestion and event construction
- Convert flow rows into semantic event sequences using a shared mapper (`src/ingestion/flow_event_mapper.py`).
- Build flow-level session IDs.

2. Baseline model training
- Train transition graph/Markov probabilities on benign baseline data.
- Save model + baseline features.

3. Hybrid anomaly scoring
- Structural sequence score (transition probabilities).
- Timing/rate regularity signals.
- Statistical outlier score (IsolationForest trained on baseline features).

4. Evaluation
- Validation-tuned threshold.
- Held-out test metrics (Precision, Recall, F1, AUC-ROC, PR-AUC).
- Bootstrap confidence intervals, calibration quality (Brier/ECE), and ablation table.
- Validation-learned hybrid component weights and automatic model selection (`full_hybrid` vs `learned_hybrid`).

5. Optional post-analysis
- Root Cause Analysis (RCA) for explanations.
- Behavioral clustering (HDBSCAN) for campaign/family discovery.

## Project Structure

```text
src/
  ingestion/
    flow_event_mapper.py
    prepare_baseline.py
    mixed_data.py
    pcap_to_sessions.py
  casual/
    build_dag.py
    anamoly_scoring.py
    evaluate_results.py
    root_cause_analysis.py
    behavioral_clustering.py
  experiments/
    run_cross_dataset_experiments.py
  run_full_pipeline.py

data/
  raw/                # Input CSV/PCAP files
  processed/          # Intermediate model/features/results
  results/            # Evaluation plots/reports
  experiments/        # Per-dataset experiment outputs
```

Note: module folder is named `casual/` in the codebase (kept as-is for compatibility).

## Setup

From project root:

```bash
cd /home/anurag/Projects/testingtest
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How To Run

### 1) End-to-end pipeline (baseline + scoring + RCA + clustering)

```bash
python src/run_full_pipeline.py
```

### 2) Evaluation only

```bash
python src/casual/evaluate_results.py
```

### 3) Cross-dataset experiments (paper benchmarking)

```bash
python -m src.experiments.run_cross_dataset_experiments
```

This writes:
- `data/experiments/cross_dataset_summary.csv`
- `data/experiments/cross_dataset_summary.md`
- per-dataset reports under `data/experiments/<experiment>/results/`

### 4) Research protocols (leakage-safe random + time split, optional cross-dataset)

```bash
python -m src.experiments.run_research_protocols --n-bootstrap 200
python -m src.experiments.run_research_protocols --include-cross-dataset --n-bootstrap 200
python -m src.experiments.run_research_protocols --n-bootstrap 200 --split-min-test-anomalies 200 --split-min-val-anomalies 200
```

This writes:
- `data/experiments/research_protocols/protocol_summary.csv`
- `data/experiments/research_protocols/protocol_summary.md`
- per-protocol reports under `data/experiments/research_protocols/<protocol>/`

### 5) Publishable suite (protocols + publication bundle)

```bash
python -m src.experiments.run_publishable_suite --n-bootstrap 200 --include-cross-dataset
```

This writes:
- `data/experiments/publication_bundle/main_metrics_table.csv`
- `data/experiments/publication_bundle/main_metrics_table.md`
- `data/experiments/publication_bundle/publication_summary.md`
- `data/experiments/publication_bundle/reproducibility_manifest.json`
- paper templates in `paper/`

## Main Outputs

- `data/processed/dag_model_complete.pkl`
  - trained transition model and transition metadata.
- `data/processed/baseline_features.csv`
  - baseline session feature space used for statistical scoring.
- `data/processed/anomaly_scores_with_features.csv`
  - session-level anomaly scores and components.
- `data/results/evaluation_report.txt`
  - metrics from current run.
- `data/results/ablation_results.csv`
  - full-hybrid vs component ablations for research reporting.
- `data/results/per_label_analysis.csv`
  - attack-family level behavior summary.
- `data/results/per_family_error_analysis.csv`
  - per-attack-family precision/recall/miss-rate at selected threshold.
- `data/results/learned_hybrid_weights.csv`
  - learned contribution of structural/intensity/regularity/statistical components.
- `data/experiments/cross_dataset_summary.csv`
  - consolidated benchmark table across datasets.

## Research Notes

- The headline evaluation is leakage-safe: one global threshold is learned on validation and applied to test.
- Label-aware family thresholds (if enabled) are oracle diagnostics, not primary research metrics.
- Attack-family performance can vary by dataset; always report per-family behavior, not only global metrics.
- For full change history during this iteration, see `update.md`.

## Current Limitations

- Web attack family performance is weaker than scan/flood families in current mapping/scoring setup.
- Some datasets are highly imbalanced (for example, very rare infiltration rows), so confidence intervals are recommended for paper reporting.
- Dataset-specific calibration may be needed for deployment scenarios.

## Next Suggested Research Steps

1. Add calibrated probability outputs and PR-AUC reporting.
2. Add ablations (Markov-only vs hybrid vs statistical-only).
3. Add confidence intervals via bootstrap across sessions.

## Paper-Ready Asset Generation

After running experiments, generate manuscript figures and a metric snapshot:

```bash
python -m src.experiments.build_paper_assets
```

Outputs:
- `paper/figures/fig1_protocol_metrics.png`
- `paper/figures/fig2_cross_dataset_f1_pr_auc.png`
- `paper/figures/fig3_rca_driver_counts.png`
- `paper/figures/fig4_rca_cluster_hints.png`
- `paper/figures/fig5_family_precision_comparison.png`
- `paper/RESULTS_SNAPSHOT_2026-03-06.md`

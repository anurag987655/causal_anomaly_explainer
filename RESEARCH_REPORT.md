# NetCausalAI Research Report (Updated)

Last updated: February 15, 2026

## 1. Current System State
NetCausalAI is now a hybrid anomaly detection pipeline that combines:
- Markov transition modeling over semantic event sequences
- Statistical outlier detection (Isolation Forest on baseline features)
- Root Cause Analysis (RCA) explanations
- Optional behavioral clustering for attack-family discovery

Core modules:
- `src/ingestion/flow_event_mapper.py`
- `src/casual/build_dag.py`
- `src/casual/anamoly_scoring.py`
- `src/casual/evaluate_results.py`
- `src/casual/root_cause_analysis.py`
- `src/casual/behavioral_clustering.py`

## 2. Key Improvements Completed
1. Fixed transition graph quality and non-interactive reliability
- Kept self-loops in DAG transitions (critical for repetitive benign behavior modeling).
- Replaced blocking `plt.show()` with `plt.close()` in DAG visualization.

2. Fixed scoring pipeline and robustness issues
- `run_anomaly_scoring_with_rca(...)` now respects caller-provided file paths.
- Added timestamp unit inference (seconds vs milliseconds) to avoid rate-scaling errors.
- Added hybrid score components (sequence + timing + statistical outlier score).
- Replaced static anomaly cutoff with percentile-based labeling (`anomaly_percentile`).

3. Improved data integrity at session construction layer
- Removed destination-port mega-session grouping.
- Switched to flow-level session IDs for label and temporal consistency.
- Propagated flow-level metadata into session rows (`session_duration`, bytes, packet rate, protocol, etc.).

4. Upgraded evaluation methodology
- Added leakage-safer evaluation in `evaluate_results.py`:
  - threshold tuned on validation split
  - final metrics reported on held-out test split
- Added fixed-threshold evaluation mode.
- Added report export to `data/results/evaluation_report.txt`.

5. Added experiment framework and stronger ingestion consistency
- Added cross-dataset runner: `src/experiments/run_cross_dataset_experiments.py`.
- Unified training/evaluation event vocabulary using shared `flow_event_mapper`.
- Added v2 semantic mapping tuning for better discriminative cues.

## 3. Latest Metrics (From Current Artifacts)
### A. Latest main evaluation report
Source: `data/results/evaluation_report.txt`

| Metric | Value |
|:--|--:|
| Precision | 0.2619 |
| Recall | 0.8193 |
| F1 | 0.3969 |
| AUC-ROC | 0.7494 |
| Threshold (validation-tuned) | 0.2278 |
| Eval split sizes | validation=70000, test=30000 |

### B. Cross-dataset summary (selected rows)
Source: `data/experiments/cross_dataset_summary.csv`

| Dataset | Precision | Recall | F1 | AUC-ROC |
|:--|--:|--:|--:|--:|
| Friday PortScan | 0.7629 | 0.9891 | 0.8614 | 0.7933 |
| Tuesday Patator | 0.5797 | 1.0000 | 0.7340 | 0.7704 |
| Wednesday DoS | 0.6044 | 0.8992 | 0.7229 | 0.6140 |
| Thursday Infiltration | 0.6667 | 0.1818 | 0.2857 | 0.8972 |
| Thursday WebAttack | 0.0802 | 1.0000 | 0.1485 | 0.4086 |
| Friday DDoS | 0.5000 | 1.0000 | 0.6667 | 0.4643 |

### C. v2 mapping deltas (targeted comparison)
Source: `data/experiments/v2_targeted_comparison.csv`

| Dataset | F1 Delta | AUC Delta |
|:--|--:|--:|
| PortScan | +0.0151 | +0.0517 |
| Wednesday DoS | +0.0562 | +0.0531 |
| Thursday WebAttack | +0.0000 | +0.2251 |

## 4. Interpretation
- The pipeline is now stable and reproducible across multiple CICIDS attack families.
- PortScan and DoS families are comparatively strong.
- Web attacks remain weak on precision despite improved ranking (AUC gain in v2), indicating threshold/calibration and feature separability issues.
- Infiltration shows strong ranking (high AUC) but weak recall at current thresholding, suggesting class-imbalance-sensitive tuning is still needed.

## 5. Research Priorities (Next)
1. Add PR-AUC and calibration metrics (Brier score, reliability plots) for imbalanced families.
2. Run explicit ablations: Markov-only vs statistical-only vs hybrid.
3. Add confidence intervals (bootstrap at session level) for all reported metrics.
4. Add per-family threshold policies or cost-sensitive thresholding.
5. Improve web-attack feature representation in `flow_event_mapper.py` (payload/HTTP-oriented cues if available).

## 6. Reproduction Commands
```bash
# End-to-end baseline + scoring + RCA + clustering
python src/run_full_pipeline.py

# Evaluation report
python src/casual/evaluate_results.py

# Cross-dataset benchmark
python -m src.experiments.run_cross_dataset_experiments
```

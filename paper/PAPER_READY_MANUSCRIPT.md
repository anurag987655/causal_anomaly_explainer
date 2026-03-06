# Leakage-Safe Behavioral Anomaly Detection for CICIDS-Style Network Flows with Root-Cause Explanations

## Abstract
Network anomaly detection studies often report optimistic results due to data leakage, weak split protocols, or missing explanation layers. We present NetCausalAI, a reproducible behavioral anomaly detection pipeline for CICIDS-style flow datasets that combines sequence structure, temporal behavior, and post-hoc root-cause analysis (RCA). The evaluation protocol uses validation-tuned thresholding and held-out test reporting under random, temporal, and cross-dataset settings. On the March 6, 2026 run configuration (temporal-only scorer with F0.5 threshold objective), NetCausalAI achieves strong in-distribution performance (random split: Precision 0.6425, Recall 0.8682, F1 0.7385, PR-AUC 0.5146), with weaker but non-trivial temporal and cross-dataset performance (time split F1 0.2948; cross-dataset macro F1 0.4967, macro PR-AUC 0.3882). RCA over the top 200 anomalous sessions shows consistent dominant mechanisms, especially timing and rare-transition violations, enabling interpretable failure analysis. Results indicate a measurable tradeoff between precision-oriented tuning and cross-domain robustness, motivating adaptive model-policy selection for deployment.

## 1. Introduction
ML-based IDS systems frequently optimize for single-split aggregate metrics, which can mask leakage and limit deployment value. For operational adoption, a detector must satisfy four requirements:

1. Leakage-safe protocol design.
2. Reproducible experiment execution.
3. Interpretable anomaly causes.
4. Honest reporting of distribution-shift failure modes.

NetCausalAI addresses these requirements by combining anomaly scoring and RCA in one pipeline. This paper focuses on protocol quality and explainability under realistic split conditions.

## 2. Contributions
1. A leakage-aware evaluation pipeline for CICIDS-style sessionized flow data, including random, temporal, and cross-dataset tracks.
2. A behavior-centric anomaly framework integrating sequence structure and temporal dynamics.
3. RCA outputs that expose dominant anomaly drivers and recurring behavior families.
4. A reproducible artifact workflow with one-command experiment and publication-bundle generation.

## 3. Method
### 3.1 Event Construction and Sessionization
Raw flow records are mapped to semantic event sequences using a shared event mapper. Session-level features and labels are retained for downstream scoring and analysis.

### 3.2 Scoring Models
The evaluation framework supports multiple score models (`full_hybrid`, `temporal_only`, `markov_only`, `statistical_only`, `hybrid_without_stat`, `learned_hybrid`). In this paper snapshot, the deployed preference is `temporal_only`.

### 3.3 Threshold Selection
A global threshold is learned on validation data only and then applied to held-out test data. For this run, the threshold objective is `F0.5`, prioritizing precision over recall relative to F1.

### 3.4 Interpretability Layer
RCA identifies high-impact transition and timing deviations per anomalous session, then groups sessions into behavior families (`cluster_hint`) to support campaign-level triage.

## 4. Experimental Setup
### 4.1 Data and Protocols
We evaluate on:

1. Random split.
2. Time split (`time(session_id_order_fallback)` when explicit timestamps are unavailable).
3. Cross-dataset benchmark over seven CICIDS-derived subsets.

### 4.2 Metrics
Primary metrics: Precision, Recall, F1, ROC-AUC, PR-AUC.  
Reliability metrics: Brier score and ECE.  
Uncertainty: bootstrap confidence intervals in per-protocol reports.

## 5. Results
### 5.1 Protocol-Level Performance
From `data/experiments/publication_bundle/main_metrics_table.csv`:

1. Random split: Precision 0.6425, Recall 0.8682, F1 0.7385, PR-AUC 0.5146.
2. Time split: Precision 0.1790, Recall 0.8340, F1 0.2948, PR-AUC 0.1358.
3. Cross-dataset macro: Precision 0.4844, Recall 0.5731, F1 0.4967, PR-AUC 0.3882.

See Figure 1: `paper/figures/fig1_protocol_metrics.png`.

### 5.2 Cross-Dataset Behavior
Best and weakest tracks (by F1):

1. Strong: `friday_afternoon_portscan` (F1 0.9181, PR-AUC 0.8059).
2. Strong: `friday_afternoon_ddos` (F1 0.7747, PR-AUC 0.6133).
3. Weak: `thursday_infiltration` (F1 0.0000, PR-AUC 0.0231).
4. Weak: `friday_morning_bot` (F1 0.2387, PR-AUC 0.1025).

See Figure 2: `paper/figures/fig2_cross_dataset_f1_pr_auc.png`.

### 5.3 Interpretability Findings (RCA)
Top driver frequencies over top-200 anomalous sessions:

1. `timing_violation`: 198
2. `rare_transition`: 127
3. `single_point_failure`: 104
4. `context_violation`: 61

Cluster hints:

1. `unusual_pattern`: 162
2. `novel_behavior`: 38

See Figure 3: `paper/figures/fig3_rca_driver_counts.png` and Figure 4: `paper/figures/fig4_rca_cluster_hints.png`.

### 5.4 Family-Level Precision Contrast
Per-family precision differs strongly across random vs time protocols, reinforcing temporal shift sensitivity. See Figure 5: `paper/figures/fig5_family_precision_comparison.png`.

## 6. Discussion
The results support a concrete tradeoff:

1. Precision-oriented tuning (`temporal_only + F0.5`) substantially improves random-split precision/F1.
2. Cross-dataset macro performance can degrade under the same policy.

This implies model/threshold policy should be selected according to deployment objective:

1. High-confidence triage (precision-priority).
2. Broad detection coverage across domains (robustness-priority).

## 7. Limitations and Threats to Validity
1. Temporal split fallback based on session-order indices may not represent real-world chronology in all files.
2. Severe class imbalance in families like infiltration amplifies threshold instability.
3. Event mapping quality directly affects separability, especially for web-attack and bot-like behaviors.
4. Cross-dataset experiments are all within the CICIDS ecosystem; external validity remains open.

## 8. Reproducibility
Main commands:

```bash
cd /home/anurag/Projects/testingtest
source venv/bin/activate
python -m src.experiments.run_research_protocols --include-cross-dataset --n-bootstrap 100 --threshold-objective f0.5 --score-model-preference temporal_only
python -m src.experiments.build_publication_bundle
python -m src.experiments.build_paper_assets
```

Key generated artifacts:

1. `data/experiments/research_protocols/protocol_summary.md`
2. `data/experiments/cross_dataset_summary.md`
3. `data/experiments/publication_bundle/main_metrics_table.md`
4. `paper/RESULTS_SNAPSHOT_2026-03-06.md`
5. `paper/figures/*.png`

## 9. Conclusion
NetCausalAI provides a reproducible and interpretable anomaly-detection workflow with leakage-aware reporting. The current evidence shows clear utility on in-distribution detection and RCA transparency, while highlighting cross-domain robustness constraints under precision-prioritized tuning. This tradeoff is a meaningful result, not a negative outcome, and should be reported explicitly in final submission versions.

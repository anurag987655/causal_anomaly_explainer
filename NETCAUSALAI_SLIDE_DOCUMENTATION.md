# NetCausalAI Slide Documentation

Prepared from project artifacts in `Projects/testingtest`  
Data snapshot: February 15, 2026

## Slide 1: Title
**Title:** NetCausalAI: Interpretable Hybrid Network Anomaly Detection  
**Subtitle:** From anomaly scoring to root-cause explanation and behavior families

**Presenter Notes**
- NetCausalAI is a research pipeline for intrusion detection on CICIDS-style flow data.
- It combines sequence modeling, statistical outlier detection, explainability, and clustering.

## Slide 2: Problem Statement
**Title:** Why NetCausalAI?

**Slide Content**
- Traditional IDS outputs often stop at a score or binary label.
- Security teams need to know:
  - How anomalous a session is
  - Why it is anomalous
  - Which anomalies belong to the same behavior family
- Goal: produce quantitative and explainable outputs suitable for research and operations.

**Presenter Notes**
- Emphasize that explainability and grouping improve actionability, not just detection.

## Slide 3: System Overview
**Title:** End-to-End Pipeline

**Slide Content**
1. Ingestion + semantic event construction (`flow_event_mapper.py`)
2. Baseline Markov transition model training
3. Hybrid scoring:
   - Structural sequence score
   - Timing/rate regularity score
   - Isolation Forest statistical score
4. Evaluation (validation-tuned threshold, held-out test metrics)
5. Post-analysis:
   - RCA explanations
   - Behavioral clustering

**Presenter Notes**
- The architecture is modular and reproducible across datasets.

## Slide 4: Architecture Diagram
**Title:** Core Components and Data Flow

**Slide Content**
- Input: CICIDS flow CSVs / PCAP-derived sessions
- Processing modules:
  - `src/ingestion/*`
  - `src/casual/build_dag.py`
  - `src/casual/anamoly_scoring.py`
  - `src/casual/evaluate_results.py`
  - `src/casual/root_cause_analysis.py`
  - `src/casual/behavioral_clustering.py`
- Key artifacts:
  - `data/processed/dag_model_complete.pkl`
  - `data/processed/anomaly_scores_with_features.csv`
  - `data/results/evaluation_report.txt`

**Visual Suggestion**
- Include `data/processed/dag_graph_complete.png`.

## Slide 5: Major Engineering Improvements
**Title:** What Was Improved (Recent Update Cycle)

**Slide Content**
- Fixed path handling and timestamp unit inference in scoring.
- Replaced leakage-prone evaluation with validation/test split.
- Replaced invalid destination-port sessionization with flow-level sessions.
- Added hybrid Markov + Isolation Forest scoring.
- Unified event vocabulary across training/evaluation.
- Added cross-dataset experiment runner.
- Tuned v2 semantic mapping with stronger discriminative cues.

**Presenter Notes**
- These changes improved reliability, reproducibility, and metric quality.

## Slide 6: Main Evaluation Snapshot
**Title:** Latest Main Metrics (`data/results/evaluation_report.txt`)

**Slide Content**
- Threshold (validation tuned): **0.2278**
- Evaluation split: validation=70,000 / test=30,000
- Test metrics:
  - Precision: **0.2619**
  - Recall: **0.8193**
  - F1: **0.3969**
  - AUC-ROC: **0.7494**

**Presenter Notes**
- Current global setting favors recall; precision still needs improvement in harder families.

## Slide 7: Cross-Dataset Results
**Title:** Generalization Across Attack Families

**Slide Content**
- Strong:
  - Friday PortScan: F1 **0.8614**, AUC **0.7933**
  - Tuesday Patator: F1 **0.7340**, AUC **0.7704**
  - Wednesday DoS: F1 **0.7229**, AUC **0.6140**
- Mixed/Weak:
  - Thursday Infiltration: F1 **0.2857**, AUC **0.8972**
  - Thursday WebAttack: F1 **0.1485**, AUC **0.4086**
  - Friday DDoS: F1 **0.6667**, AUC **0.4643**

**Presenter Notes**
- Infiltration has good ranking but low recall at current threshold.
- Web attacks remain the hardest family and drive false positives.

## Slide 8: v2 Mapping Impact
**Title:** Targeted Improvements from v2 Semantic Mapping

**Slide Content**
- PortScan:
  - F1 delta: **+0.0151**
  - AUC delta: **+0.0517**
- Wednesday DoS:
  - F1 delta: **+0.0562**
  - AUC delta: **+0.0531**
- Thursday WebAttack:
  - F1 delta: **+0.0000**
  - AUC delta: **+0.2251**

**Presenter Notes**
- v2 improved ranking signal on weak classes (especially WebAttack AUC), but threshold/calibration is still limiting F1.

## Slide 9: Explainability and RCA
**Title:** From Scores to Actionable Explanations

**Slide Content**
- RCA outputs identify likely anomaly drivers per session.
- Outputs:
  - `data/processed/rca_explanations.csv`
  - `data/processed/rca_explanations.json`
  - `data/processed/rca_summary.txt`
- Benefit: analyst can prioritize investigation by causal cues, not only by score.

**Presenter Notes**
- Position this as a key differentiator from black-box anomaly detectors.

## Slide 10: Behavioral Clustering
**Title:** Grouping Anomalies into Behavior Families

**Slide Content**
- Clustering module groups anomalous sessions by behavior similarity.
- Outputs:
  - `data/processed/clustering_results.csv`
  - `data/processed/cluster_interpretations.json`
  - `data/processed/cluster_visualization.png`
- Benefit: campaign-level triage and pattern discovery.

**Presenter Notes**
- Show how clustering reduces analyst load by moving from individual alerts to grouped campaigns.

## Slide 11: Limitations and Risks
**Title:** Current Gaps

**Slide Content**
- Precision remains low in WebAttack/Bot-like families.
- Class imbalance affects threshold stability (notably infiltration).
- One global threshold does not fit all families equally.
- Calibration and uncertainty estimates are not yet fully integrated.

**Presenter Notes**
- State this clearly to keep claims credible and publication-ready.

## Slide 12: Research Roadmap
**Title:** Next 12-Week Plan

**Slide Content**
1. Add calibration layer + PR-AUC/Brier/ECE reporting.
2. Run full ablation matrix:
   - Markov-only
   - Statistical-only
   - Hybrid
   - Hybrid + calibration
   - Hybrid + calibration + hard-negative mining
3. Add leave-one-family-out and leave-one-day-out robustness splits.
4. Add confidence intervals + significance testing.
5. Deliver submission-grade reproducibility package.

**Presenter Notes**
- This turns the current strong prototype into tier-1 publication evidence.

## Slide 13: Closing
**Title:** Key Takeaways

**Slide Content**
- NetCausalAI is a working, explainable hybrid anomaly detection stack.
- It already generalizes well on multiple families (PortScan/DoS/Patator).
- Main priority is precision uplift for weak families through calibration and threshold policy.
- The project is structured for reproducible research progression.

**Presenter Notes**
- End by inviting questions on method, metrics, and deployment strategy.

## Appendix: Optional Visuals to Insert
- `data/processed/dag_graph_complete.png`
- `data/results/confusion_matrix_optimal.png`
- `data/results/score_distribution.png`
- `data/processed/cluster_visualization.png`
- `data/experiments/v2_portscan/results/confusion_matrix_optimal.png`
- `data/experiments/v2_wednesday_dos/results/confusion_matrix_optimal.png`
- `data/experiments/v2_thursday_webattack/results/confusion_matrix_optimal.png`

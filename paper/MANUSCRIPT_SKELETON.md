# NetCausalAI Manuscript Skeleton

## Title
Leakage-Safe Hybrid Behavioral Anomaly Detection for CICIDS-Style Network Flows

## Abstract
- Problem:
- Method:
- Protocols:
- Main quantitative results:
- Practical implications:

## 1. Introduction
- Problem setting and motivation.
- Why leakage-safe IDS evaluation matters.
- Contributions:
  - leakage-safe evaluation protocol;
  - hybrid behavioral scorer;
  - reproducible benchmark suite.

## 2. Related Work
- Classical IDS and anomaly scoring.
- Sequence/graph-based behavior modeling.
- Calibration and thresholding under class imbalance.
- Evaluation leakage pitfalls in security ML.

## 3. Method
### 3.1 Data and Event Mapping
### 3.2 Markov + Statistical Hybrid Scoring
### 3.3 Validation-Tuned Global Thresholding
### 3.4 Optional Oracle Diagnostics (Non-headline)

## 4. Experimental Setup
### 4.1 Datasets
### 4.2 Protocols
- Random split
- Time split (non-degenerate constraints)
- Cross-dataset split
### 4.3 Metrics
- Precision, Recall, F1, PR-AUC, ROC-AUC
- Brier score, ECE
- Bootstrap 95% CI

## 5. Results
### 5.1 Main Quantitative Table
- Use `data/experiments/publication_bundle/main_metrics_table.csv`.
### 5.2 Cross-Dataset Results
### 5.3 Weak-Family Behavior
### 5.4 Calibration and Reliability

## 6. Ablation Study
- markov_only vs statistical_only vs hybrid variants.
- discuss tradeoffs and confidence intervals.

## 7. Interpretability
- RCA case studies.
- behavior family clustering examples.

## 8. Limitations and Threats to Validity
- dataset bias and class imbalance.
- synthetic session-order fallback in temporal split.
- external validity and deployment constraints.

## 9. Reproducibility Statement
- command(s) used;
- code and artifact paths;
- environment manifest.

## 10. Conclusion
- findings, limitations, and future work.

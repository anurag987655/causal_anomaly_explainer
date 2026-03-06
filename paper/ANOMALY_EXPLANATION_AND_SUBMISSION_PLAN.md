# NetCausalAI: Anomaly Explanation and Submission Plan

## 1) What The Network Anomaly Actually Is

From `data/processed/rca_explanations.csv` and `data/processed/rca_summary.txt` (200 high-score anomalous sessions):

- RCA family split:
  - `unusual_pattern`: 162 / 200 (81.0%)
  - `novel_behavior`: 38 / 200 (19.0%)
- Most common anomaly drivers:
  - `timing_violation`: 198 / 200
  - `rare_transition`: 127 / 200
  - `single_point_failure`: 104 / 200
  - `context_violation`: 61 / 200
- Most repeated root causes:
  - `MICROBURST -> PACKET_FLOOD_HINT too fast`: 145 sessions
  - `TCP_SYN_START -> AUTH_SERVICE statistically improbable`: 101 sessions
  - `TCP_FLOW_START -> MAIL_SERVICE novel transition`: 38 sessions

Interpretation:
- The dominant anomaly pattern is not random noise.
- It is a consistent behavioral shift where session transition order and timing differ from learned baseline behavior.
- In plain terms: many sessions look "too fast" or "structurally unusual" relative to normal traffic, especially around SYN/start and burst/flood-like transitions.

## 2) Why Results Feel Confusing

The model ranks anomalies reasonably well but thresholded precision is weak in harder settings:

- Random split (`data/experiments/research_protocols/random_split/evaluation_report.txt`):
  - Precision: 0.4198
  - Recall: 0.9554
  - F1: 0.5833
- Time split (`data/experiments/research_protocols/time_split/evaluation_report.txt`):
  - Precision: 0.1351
  - Recall: 0.9630
  - F1: 0.2369

This is why it feels like "too many contradictory results":
- high recall means anomalies are usually caught,
- low precision means many benign sessions are also flagged.

Additional signal that thresholding/model mix needs tuning:
- In ablations, `temporal_only` outperforms `full_hybrid` on F1 for both random and time protocols (`ablation_results.csv`).

## 3) Is This Publishable?

Yes, if positioned honestly as:
- strong detection sensitivity and ranking;
- weak precision under temporal shift / class imbalance;
- interpretable anomaly signatures via RCA + clustering;
- clear failure analysis and ablation-backed next steps.

You already have publication artifacts generated on **2026-03-04**:
- `data/experiments/publication_bundle/main_metrics_table.md`
- `data/experiments/publication_bundle/publication_summary.md`
- `data/experiments/research_protocols/protocol_summary.md`

`publication_summary.md` reports **7/7 claim checks passed**.

## 4) Minimal Submission Story (Use This)

1. Problem:
Leakage-safe network anomaly detection with interpretable root causes is hard under imbalance and temporal drift.

2. Method:
Hybrid sequence + statistical scoring with validation-only thresholding, plus RCA and behavioral clustering.

3. Main findings:
- random split F1 = 0.5833, recall = 0.9554
- time split F1 = 0.2369, recall = 0.9630
- cross-dataset macro F1 = 0.6173
- dominant anomalies are timing/context transition violations (not arbitrary outliers)

4. Honest limitation:
Precision drops strongly on time split and some families, indicating calibration and feature-separability limits.

5. Contribution:
Reproducible leakage-safe protocol + interpretable failure modes + clear roadmap for precision recovery.

## 5) Immediate Next Actions (No Reinvention)

1. Use `paper/MANUSCRIPT_SKELETON.md` and fill Sections 5, 7, 8 directly from current artifacts.
2. Add one table: top RCA drivers/frequencies from `rca_explanations.csv` (counts above).
3. Keep headline metrics leakage-safe only (no oracle thresholds as headline).
4. Submit as workshop/short paper first (method + interpretability + rigorous failure analysis).

## 6) Decision

Do not quit based on these results.

The project already has:
- reproducible pipeline,
- publishable protocol outputs,
- and a defensible scientific story with both strengths and limitations.

The correct move is scoped submission, not abandonment.

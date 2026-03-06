# NetCausalAI Tier-1 Research Roadmap

Date: February 15, 2026
Owner: Anurag
Project: NetCausalAI (`/home/anurag/Projects/testingtest`)

## 1) Research Objective

Build a family-robust, interpretable network anomaly detection system that sustains high precision under imbalance and domain shift, with reproducible evidence suitable for a tier-1 research submission.

## 2) Current Baseline Snapshot

Current strengths:
- Strong precision on some scan-like families (PortScan).
- End-to-end pipeline already exists (ingestion -> scoring -> evaluation -> experiments).
- Explainability artifacts and cross-dataset experiment flow are already implemented.

Current gaps:
- Low precision in select families (notably WebAttack/Bot in current setup).
- One-model threshold behavior does not uniformly generalize across families.
- Need stronger calibration, family-aware hard-negative handling, and stricter evaluation protocol.

## 3) Success Criteria (Tier-1 Readiness)

Minimum acceptance targets:
1. Macro-precision improvement >= 20% over frozen baseline.
2. Per-family precision uplift in weakest families without severe recall collapse.
3. Demonstrated robustness under domain shift:
- Leave-one-family-out
- Leave-one-day-out
4. Statistically supported claims:
- 95% confidence intervals
- significance testing for key comparisons
5. Full reproducibility:
- fixed seeds
- one-command rerun
- versioned artifacts and config registry

## 4) Research Questions and Hypotheses

RQ1: Can calibrated hybrid scoring improve precision across heterogeneous attack families?
- H1: A meta-calibration layer on top of hybrid components will improve macro precision and PR-AUC.

RQ2: Can hard-negative mining reduce false positives in weak families?
- H2: Iterative false-positive mining will significantly improve precision in sparse/overlapping families.

RQ3: Does family-aware thresholding improve operational trade-offs?
- H3: Validation-driven per-family operating points outperform a single global threshold on macro metrics.

RQ4: Do gains persist under distribution shift?
- H4: Calibration + hard-negative loops improve leave-one-family/day generalization.

## 5) Method Plan

### 5.1 Model Stack

Base layer:
- Existing hybrid score components:
  - structural score
  - intensity score
  - regularity score
  - statistical score

Calibration layer:
- Train a lightweight meta-model (e.g., logistic regression / gradient boosting) on validation split only.
- Inputs:
  - hybrid score components
  - core flow/session features
- Output:
  - calibrated anomaly probability

Uncertainty layer:
- Bootstrap or small ensemble for confidence intervals and stability estimates.

### 5.2 Feature Expansion (All Families)

Implement and validate feature families for:
1. Recon/Scan:
- destination port fan-out
- SYN-dominance trends
- short-session scan burst signatures
2. DoS/DDoS/Flood:
- packet rate escalation
- concurrency bursts
- sustained high-volume monotony indicators
3. Credential abuse (Patator-like):
- repeated short failed-attempt patterns
- service-target repetition
4. Web attacks:
- request/response asymmetry cues (if available)
- bursty low-byte repetitive request patterns
5. Bot/C2-like:
- periodic beacon timing
- regular small-payload outbound patterns

### 5.3 Hard-Negative Mining Loop

Per iteration:
1. Run model on validation/test candidates.
2. Collect top false positives by family and confidence.
3. Add curated hard negatives into training schedule.
4. Retrain and evaluate.

Stop criteria:
- precision gain < 1% over 2 consecutive loops, or
- recall degradation exceeds policy limit.

### 5.4 Thresholding Policy

1. Keep global calibrated score for ranking.
2. Select operating points from validation only:
- global threshold (default)
- optional family-specific thresholds for operational mode
3. Report strict/balanced/high-recall operating modes.

## 6) Evaluation Protocol

Primary metrics:
- Per-family: Precision, Recall, F1, PR-AUC, AUROC
- Global: macro and micro variants
- Calibration: ECE, Brier score
- Operational: false positives per 10k sessions

Robustness splits:
1. In-distribution holdout
2. Leave-one-family-out
3. Leave-one-day-out (temporal shift)

Statistical rigor:
1. Bootstrap confidence intervals
2. Paired significance tests across model variants

## 7) Ablation Matrix (Publication Critical)

Required ablations:
1. Markov-only
2. Statistical-only
3. Hybrid (current)
4. Hybrid + calibration
5. Hybrid + calibration + hard-negative mining
6. Hybrid + calibration + hard-negative + family-threshold policy

Output:
- one unified table + per-family delta chart

## 8) Explainability Validation

RCA is kept, but validate quality:
1. Explanation fidelity (does removing cited drivers reduce score?)
2. Stability (consistency under small perturbations)
3. Analyst utility review (error taxonomy and actionability)

## 9) Engineering and Reproducibility

Mandatory engineering standards:
1. Config-driven experiments (`configs/`):
- dataset split config
- model config
- threshold policy config
2. Versioned artifacts:
- model hash
- dataset version fingerprint
- metrics bundle
3. One-command rerun script:
- `scripts/run_research_suite.sh`
4. Seed control:
- numpy/sklearn/random seeds fixed

## 10) 12-Week Execution Timeline

Week 1:
1. Freeze baseline.
2. Create config registry.
3. Standardize reporting schema.

Week 2:
1. Implement calibration layer.
2. Add calibration metrics (ECE/Brier).
3. Run initial comparison vs baseline.

Week 3:
1. Add uncertainty estimates.
2. Add confidence interval reporting.

Week 4:
1. Feature expansion for Scan/DoS families.
2. Re-evaluate robustness splits.

Week 5:
1. Feature expansion for Web/Bot/Credential families.
2. Drift diagnostics pass.

Week 6:
1. Hard-negative mining loop v1.
2. Full evaluation and error taxonomy.

Week 7:
1. Hard-negative mining loop v2.
2. Family-threshold policy experiments.

Week 8:
1. Complete ablation matrix.
2. Generate comparative plots/tables.

Week 9:
1. Significance testing and CI audit.
2. Reproducibility dry run from clean workspace.

Week 10:
1. Draft method and experiment sections.
2. Build figure pack.

Week 11:
1. Draft results/discussion/limitations.
2. Internal review and gap fixing.

Week 12:
1. Final reproducibility package.
2. Submission-ready manuscript and appendix.

## 11) Immediate Post-Demo Next Actions

1. Freeze today's baseline artifacts and summary table.
2. Implement calibration layer first (highest cross-family impact).
3. Add robust evaluation script for leave-one-family/day splits.
4. Start hard-negative mining with weakest two families first, then generalize.

## 12) Risk Register and Mitigations

Risk: precision gains hurt recall too much.
- Mitigation: enforce recall floor in threshold selection policy.

Risk: family-specific overfitting.
- Mitigation: strict leave-one-family/day evaluation and cross-validation.

Risk: noisy labels in minority classes.
- Mitigation: uncertainty-aware sampling and manual audit of top-error slices.

Risk: reproducibility drift.
- Mitigation: artifact hashing + fixed configs + seed lock + scripted runs.

## 13) Deliverables Checklist

Technical:
1. Calibrated hybrid model implementation.
2. Hard-negative mining pipeline.
3. Family-aware evaluation suite.
4. Ablation and robustness scripts.

Research:
1. Tier-1 metric tables with CI/significance.
2. Per-family error analysis and explanation validation.
3. Reproducibility bundle and manuscript draft.


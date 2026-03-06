# NetCausalAI Research Objective (March 3, 2026)

## Primary Objective
Improve **cross-dataset anomaly detection quality** for weak attack families (especially WebAttack and Infiltration) while preserving strong performance on PortScan/DoS.

## Research Question
Can label-normalized ingestion + family-adaptive threshold calibration + targeted event-feature improvements increase macro-level performance and weak-family recall/precision on CICIDS-style datasets?

## Success Criteria (Publishable Targets)
1. Macro-F1 improvement of at least **+0.08** over current baseline in cross-dataset evaluation.
2. Macro-PR-AUC improvement of at least **+0.07**.
3. WebAttack F1 improvement of at least **+0.15**.
4. Infiltration recall improvement of at least **+0.20** without catastrophic precision collapse.
5. Results validated with bootstrap 95% confidence intervals and fixed random seeds.

## Baseline Definition
Baseline is the current pipeline with:
1. Shared event mapping.
2. Hybrid scoring + learned meta model.
3. Global validation-tuned threshold.
4. Existing cross-dataset runner output in `data/experiments/cross_dataset_summary.csv`.

## Method Variants To Test
1. `V0_BASELINE`
- Current pipeline (for reproducibility anchor).

2. `V1_LABEL_NORM + FAMILY_THRESH`
- Use normalized labels and per-family threshold policy learned on validation split.

3. `V2_WEB_FEATURES`
- Extend `flow_event_mapper.py` with web-sensitive behavioral cues.
- Keep family-threshold policy enabled.

4. `V3_HARD_NEGATIVE_MINING` (optional if time allows)
- Re-train meta layer with increased weight for top benign false positives.

## Evaluation Protocol
1. Main protocol: cross-dataset benchmarking via `src/experiments/run_cross_dataset_experiments.py`.
2. Report metrics per dataset and macro average:
- Precision, Recall, F1, ROC-AUC, PR-AUC.
3. Include weak-family table:
- WebAttack, Infiltration, Bot.
4. Include calibration:
- Brier score, ECE.
5. Include uncertainty:
- Bootstrap 95% CI.

## Publication-Ready Artifacts
1. Main comparison table: `V0` vs `V1` vs `V2` (and `V3` if available).
2. Per-family error table.
3. PR curves and calibration curves.
4. One RCA case study figure for interpretability.

## Immediate Execution Plan (Start Now)
1. Re-run baseline and save artifact snapshot (`V0`).
2. Re-run with family-threshold policy (`V1`) and compare.
3. Implement targeted web cues and run `V2`.
4. Build final tables/plots and draft paper claims only from statistically supported gains.

## Run Commands
```bash
# 1) Full pipeline
python src/run_full_pipeline.py

# 2) Cross-dataset benchmark
python -m src.experiments.run_cross_dataset_experiments
```

## Logging Rules
1. Fix random seed (`42`) for comparability.
2. Keep a run tag (`V0`, `V1`, `V2`, `V3`) in output folder names.
3. Do not overwrite summary files without retaining prior variant outputs.

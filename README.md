# NetCausalAI

NetCausalAI is a network anomaly detection project for research.

It does 4 main things:
1. Builds behavior rules from normal traffic.
2. Scores each session for anomaly risk.
3. Explains anomalies with root-cause analysis (RCA).
4. Runs evaluation protocols for publishable results.

## Project Goal

Make anomaly detection results easier to trust and explain, not just high scores.

## Quick Setup

```bash
cd /home/anurag/Projects/testingtest
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Main Commands

### 1) Run full pipeline
```bash
python src/run_full_pipeline.py
```

### 2) Run evaluation only
```bash
python src/casual/evaluate_results.py
```

### 3) Run research protocols (random + time)
```bash
python -m src.experiments.run_research_protocols --n-bootstrap 100
```

### 4) Include cross-dataset benchmark
```bash
python -m src.experiments.run_research_protocols --include-cross-dataset --n-bootstrap 100
```

### 5) Build publication bundle
```bash
python -m src.experiments.build_publication_bundle
```

### 6) Build paper figures + snapshot
```bash
python -m src.experiments.build_paper_assets
```

## Key Outputs

1. `data/experiments/research_protocols/protocol_summary.md`
2. `data/experiments/cross_dataset_summary.md`
3. `data/experiments/publication_bundle/main_metrics_table.md`
4. `paper/PAPER_READY_MANUSCRIPT.md`
5. `paper/figures/` (paper figures)

## Important Notes

1. Use leakage-safe reporting: tune threshold on validation only, then report test.
2. Treat label-aware family thresholds as diagnostics, not headline results.
3. Be explicit about weak families (for example infiltration/web attack).

## Minimal Project Layout

```text
src/
  ingestion/
  casual/
  experiments/
  run_full_pipeline.py

data/
  raw/                # input datasets (ignored in git)
  processed/          # generated intermediate files (ignored)
  experiments/        # protocol and benchmark outputs (ignored)

paper/
  PAPER_READY_MANUSCRIPT.md
  FIGURE_CAPTIONS.md
  RESULTS_SNAPSHOT_2026-03-06.md
  figures/
```

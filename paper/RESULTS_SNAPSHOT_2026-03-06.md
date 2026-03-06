# NetCausalAI Results Snapshot (2026-03-06)

## Headline Metrics

- Random split: precision=0.6425, recall=0.8682, f1=0.7385, pr_auc=0.5146
- Time split: precision=0.1790, recall=0.8340, f1=0.2948, pr_auc=0.1358
- Cross-dataset macro: precision=0.4844, recall=0.5731, f1=0.4967, pr_auc=0.3882

## Cross-Dataset Best and Weakest

Top 3 by F1:
- friday_afternoon_portscan: f1=0.9181, precision=0.8758, recall=0.9647, pr_auc=0.8059
- friday_afternoon_ddos: f1=0.7747, precision=0.7121, recall=0.8493, pr_auc=0.6133
- wednesday_dos: f1=0.5924, precision=0.7387, recall=0.4945, pr_auc=0.5873

Bottom 3 by F1:
- thursday_infiltration: f1=0.0000, precision=0.0000, recall=0.0000, pr_auc=0.0231
- friday_morning_bot: f1=0.2387, precision=0.1947, recall=0.3085, pr_auc=0.1025
- thursday_webattack: f1=0.4163, precision=0.2700, recall=0.9083, pr_auc=0.1550

## RCA Summary (Top-200 Anomalies)

- Driver `timing_violation`: 198 sessions
- Driver `rare_transition`: 127 sessions
- Driver `single_point_failure`: 104 sessions
- Driver `context_violation`: 61 sessions

- Cluster hint `unusual_pattern`: 162 sessions
- Cluster hint `novel_behavior`: 38 sessions

## Figure Inventory

- `paper/figures/fig1_protocol_metrics.png`
- `paper/figures/fig2_cross_dataset_f1_pr_auc.png`
- `paper/figures/fig3_rca_driver_counts.png`
- `paper/figures/fig4_rca_cluster_hints.png`
- `paper/figures/fig5_family_precision_comparison.png`

## Data Sources

- `data/experiments/publication_bundle/main_metrics_table.csv`
- `data/experiments/cross_dataset_summary.csv`
- `data/processed/rca_explanations.csv`
- `data/processed/rca_cluster_hints.csv`

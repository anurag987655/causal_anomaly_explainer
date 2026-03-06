"""
Build paper-ready assets (figures + metric snapshot) from current experiment outputs.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
OUT_MD = PROJECT_ROOT / "paper" / "RESULTS_SNAPSHOT_2026-03-06.md"

MAIN_TABLE_CSV = PROJECT_ROOT / "data" / "experiments" / "publication_bundle" / "main_metrics_table.csv"
CROSS_DATASET_CSV = PROJECT_ROOT / "data" / "experiments" / "cross_dataset_summary.csv"
RCA_EXPLANATIONS_CSV = PROJECT_ROOT / "data" / "processed" / "rca_explanations.csv"
RCA_CLUSTER_HINTS_CSV = PROJECT_ROOT / "data" / "processed" / "rca_cluster_hints.csv"
RANDOM_FAMILY_ERR_CSV = PROJECT_ROOT / "data" / "experiments" / "research_protocols" / "random_split" / "per_family_error_analysis.csv"
TIME_FAMILY_ERR_CSV = PROJECT_ROOT / "data" / "experiments" / "research_protocols" / "time_split" / "per_family_error_analysis.csv"


def _safe_literal_list(raw: str) -> list[str]:
    if not isinstance(raw, str) or raw.strip() == "":
        return []
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, list):
            return [str(x) for x in value]
    except Exception:
        pass
    return []


def _make_protocol_overview(main_df: pd.DataFrame) -> Path:
    plot_df = main_df[main_df["track"].isin(["random_split", "time_split", "cross_dataset_macro"])].copy()
    plot_df = plot_df.dropna(subset=["precision", "recall", "f1", "pr_auc"], how="all")
    if plot_df.empty:
        raise RuntimeError("No protocol rows found for protocol overview figure.")

    metric_cols = ["precision", "recall", "f1", "pr_auc"]
    labels = [x.replace("_", "\n") for x in plot_df["track"].tolist()]
    x = range(len(labels))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    plt.figure(figsize=(10, 5))
    for idx, m in enumerate(metric_cols):
        values = plot_df[m].fillna(0.0).to_numpy()
        plt.bar([i + offsets[idx] for i in x], values, width=width, label=m.upper())

    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Protocol-Level Metrics")
    plt.xticks(list(x), labels)
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "fig1_protocol_metrics.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def _make_cross_dataset_overview(cross_df: pd.DataFrame) -> Path:
    plot_df = cross_df.copy()
    plot_df = plot_df.sort_values(by="f1", ascending=False)

    names = plot_df["experiment"].tolist()
    x = range(len(names))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], plot_df["f1"].fillna(0.0), width=width, label="F1")
    plt.bar([i + width / 2 for i in x], plot_df["pr_auc"].fillna(0.0), width=width, label="PR-AUC")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Cross-Dataset Per-Experiment Performance")
    plt.xticks(list(x), names, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "fig2_cross_dataset_f1_pr_auc.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def _make_rca_driver_distribution(rca_df: pd.DataFrame) -> tuple[Path, Counter]:
    counter: Counter = Counter()
    for raw in rca_df["drivers"]:
        for d in _safe_literal_list(raw):
            counter[d] += 1

    top = counter.most_common(8)
    if not top:
        raise RuntimeError("No RCA drivers found.")

    names = [k for k, _ in top]
    values = [v for _, v in top]

    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.ylabel("Count")
    plt.title("Top RCA Drivers (Top-200 Anomalies)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    out = FIG_DIR / "fig3_rca_driver_counts.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out, counter


def _make_rca_cluster_distribution(cluster_df: pd.DataFrame) -> tuple[Path, Counter]:
    counter = Counter(cluster_df["cluster_hint"].astype(str).tolist())
    names = list(counter.keys())
    values = list(counter.values())

    plt.figure(figsize=(7, 5))
    plt.bar(names, values)
    plt.ylabel("Sessions")
    plt.title("RCA Cluster Hints Distribution")
    plt.tight_layout()

    out = FIG_DIR / "fig4_rca_cluster_hints.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out, counter


def _make_family_precision_comparison(random_df: pd.DataFrame, time_df: pd.DataFrame) -> Path:
    rand = random_df[["attack_family", "precision"]].rename(columns={"precision": "precision_random"})
    tim = time_df[["attack_family", "precision"]].rename(columns={"precision": "precision_time"})
    merged = pd.merge(rand, tim, on="attack_family", how="outer").fillna(0.0)
    merged = merged.sort_values(by=["precision_random", "precision_time"], ascending=False)

    names = merged["attack_family"].tolist()
    x = range(len(names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], merged["precision_random"], width=width, label="Random Split")
    plt.bar([i + width / 2 for i in x], merged["precision_time"], width=width, label="Time Split")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Precision")
    plt.title("Per-Family Precision by Protocol")
    plt.xticks(list(x), names, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "fig5_family_precision_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def _write_snapshot(
    main_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    driver_counter: Counter,
    cluster_counter: Counter,
    figure_paths: list[Path],
) -> None:
    main_lookup = {row["track"]: row for _, row in main_df.iterrows()}
    r = main_lookup.get("random_split", {})
    t = main_lookup.get("time_split", {})
    c = main_lookup.get("cross_dataset_macro", {})

    worst_cross = cross_df.sort_values(by="f1", ascending=True).head(3)
    best_cross = cross_df.sort_values(by="f1", ascending=False).head(3)

    lines = []
    lines.append("# NetCausalAI Results Snapshot (2026-03-06)")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append(f"- Random split: precision={r.get('precision', float('nan')):.4f}, recall={r.get('recall', float('nan')):.4f}, f1={r.get('f1', float('nan')):.4f}, pr_auc={r.get('pr_auc', float('nan')):.4f}")
    lines.append(f"- Time split: precision={t.get('precision', float('nan')):.4f}, recall={t.get('recall', float('nan')):.4f}, f1={t.get('f1', float('nan')):.4f}, pr_auc={t.get('pr_auc', float('nan')):.4f}")
    lines.append(f"- Cross-dataset macro: precision={c.get('precision', float('nan')):.4f}, recall={c.get('recall', float('nan')):.4f}, f1={c.get('f1', float('nan')):.4f}, pr_auc={c.get('pr_auc', float('nan')):.4f}")
    lines.append("")
    lines.append("## Cross-Dataset Best and Weakest")
    lines.append("")
    lines.append("Top 3 by F1:")
    for _, row in best_cross.iterrows():
        lines.append(f"- {row['experiment']}: f1={row['f1']:.4f}, precision={row['precision']:.4f}, recall={row['recall']:.4f}, pr_auc={row['pr_auc']:.4f}")
    lines.append("")
    lines.append("Bottom 3 by F1:")
    for _, row in worst_cross.iterrows():
        lines.append(f"- {row['experiment']}: f1={row['f1']:.4f}, precision={row['precision']:.4f}, recall={row['recall']:.4f}, pr_auc={row['pr_auc']:.4f}")
    lines.append("")
    lines.append("## RCA Summary (Top-200 Anomalies)")
    lines.append("")
    for k, v in driver_counter.most_common(5):
        lines.append(f"- Driver `{k}`: {v} sessions")
    lines.append("")
    for k, v in cluster_counter.most_common():
        lines.append(f"- Cluster hint `{k}`: {v} sessions")
    lines.append("")
    lines.append("## Figure Inventory")
    lines.append("")
    for p in figure_paths:
        lines.append(f"- `{p.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append(f"- `{MAIN_TABLE_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- `{CROSS_DATASET_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- `{RCA_EXPLANATIONS_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- `{RCA_CLUSTER_HINTS_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def build_paper_assets() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(MAIN_TABLE_CSV)
    cross_df = pd.read_csv(CROSS_DATASET_CSV)
    rca_df = pd.read_csv(RCA_EXPLANATIONS_CSV)
    cluster_df = pd.read_csv(RCA_CLUSTER_HINTS_CSV)
    random_family_df = pd.read_csv(RANDOM_FAMILY_ERR_CSV)
    time_family_df = pd.read_csv(TIME_FAMILY_ERR_CSV)

    fig1 = _make_protocol_overview(main_df)
    fig2 = _make_cross_dataset_overview(cross_df)
    fig3, driver_counter = _make_rca_driver_distribution(rca_df)
    fig4, cluster_counter = _make_rca_cluster_distribution(cluster_df)
    fig5 = _make_family_precision_comparison(random_family_df, time_family_df)

    _write_snapshot(
        main_df=main_df,
        cross_df=cross_df,
        driver_counter=driver_counter,
        cluster_counter=cluster_counter,
        figure_paths=[fig1, fig2, fig3, fig4, fig5],
    )

    print(f"Saved figures to: {FIG_DIR}")
    print(f"Saved snapshot: {OUT_MD}")


if __name__ == "__main__":
    build_paper_assets()

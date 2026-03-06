"""
Build a publication-ready bundle from experiment outputs.

Outputs:
- data/experiments/publication_bundle/publication_summary.md
- data/experiments/publication_bundle/main_metrics_table.csv
- data/experiments/publication_bundle/main_metrics_table.md
- data/experiments/publication_bundle/reproducibility_manifest.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiments"
PROTOCOL_SUMMARY = DATA_DIR / "research_protocols" / "protocol_summary.csv"
CROSS_DATASET_SUMMARY = DATA_DIR / "cross_dataset_summary.csv"
OUT_DIR = DATA_DIR / "publication_bundle"


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _to_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _extract_protocol_rows(protocol_df: Optional[pd.DataFrame]) -> List[Dict]:
    if protocol_df is None or protocol_df.empty:
        return []
    rows: List[Dict] = []
    for _, r in protocol_df.iterrows():
        rows.append(
            {
                "track": str(r.get("protocol", "")),
                "split_strategy": str(r.get("effective_split_strategy") or r.get("split_strategy") or ""),
                "precision": _to_float(r.get("precision")),
                "recall": _to_float(r.get("recall")),
                "f1": _to_float(r.get("f1")),
                "auc_roc": _to_float(r.get("auc_roc")),
                "pr_auc": _to_float(r.get("pr_auc")),
                "threshold": _to_float(r.get("threshold")),
            }
        )
    return rows


def _extract_cross_dataset_rows(cross_df: Optional[pd.DataFrame]) -> List[Dict]:
    if cross_df is None or cross_df.empty:
        return []
    rows: List[Dict] = []
    macro_row = {
        "track": "cross_dataset_macro",
        "split_strategy": "cross_dataset",
        "precision": _to_float(cross_df["precision"].mean()) if "precision" in cross_df.columns else None,
        "recall": _to_float(cross_df["recall"].mean()) if "recall" in cross_df.columns else None,
        "f1": _to_float(cross_df["f1"].mean()) if "f1" in cross_df.columns else None,
        "auc_roc": _to_float(cross_df["auc_roc"].mean()) if "auc_roc" in cross_df.columns else None,
        "pr_auc": _to_float(cross_df["pr_auc"].mean()) if "pr_auc" in cross_df.columns else None,
        "threshold": None,
    }
    rows.append(macro_row)
    return rows


def _claim_checks(table: pd.DataFrame) -> Dict[str, str]:
    checks: Dict[str, str] = {}

    random_row = table[table["track"] == "random_split"]
    time_row = table[table["track"] == "time_split"]
    cross_row = table[table["track"] == "cross_dataset_macro"]

    def val(row: pd.DataFrame, col: str) -> Optional[float]:
        if row.empty:
            return None
        return _to_float(row.iloc[0].get(col))

    random_f1 = val(random_row, "f1")
    time_f1 = val(time_row, "f1")
    cross_f1 = val(cross_row, "f1")
    cross_pr = val(cross_row, "pr_auc")

    checks["has_random_split"] = "pass" if not random_row.empty else "fail"
    checks["has_time_split"] = "pass" if not time_row.empty else "fail"
    checks["has_cross_dataset_summary"] = "pass" if not cross_row.empty else "fail"
    checks["random_split_f1_ge_0_55"] = "pass" if random_f1 is not None and random_f1 >= 0.55 else "fail"
    checks["time_split_f1_ge_0_20"] = "pass" if time_f1 is not None and time_f1 >= 0.20 else "fail"
    checks["cross_macro_f1_ge_0_60"] = "pass" if cross_f1 is not None and cross_f1 >= 0.60 else "fail"
    checks["cross_macro_pr_auc_ge_0_40"] = "pass" if cross_pr is not None and cross_pr >= 0.40 else "fail"
    return checks


def _environment_manifest() -> Dict:
    py_version = platform.python_version()
    platform_name = platform.platform()
    pip_freeze = ""
    try:
        pip_freeze = subprocess.check_output(
            ["python", "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pip_freeze = "unavailable"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": py_version,
        "platform": platform_name,
        "pip_freeze": pip_freeze.splitlines()[:2000],  # keep bounded
    }


def build_publication_bundle() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protocol_df = _safe_read_csv(PROTOCOL_SUMMARY)
    cross_df = _safe_read_csv(CROSS_DATASET_SUMMARY)

    rows = _extract_protocol_rows(protocol_df) + _extract_cross_dataset_rows(cross_df)
    if not rows:
        raise RuntimeError("No experiment summaries found. Run research protocols first.")

    table = pd.DataFrame(rows)
    table_csv = OUT_DIR / "main_metrics_table.csv"
    table_md = OUT_DIR / "main_metrics_table.md"
    table.to_csv(table_csv, index=False)
    try:
        table.to_markdown(table_md, index=False)
    except Exception:
        with open(table_md, "w") as f:
            f.write("# Main Metrics Table\n\n")
            f.write(table.to_csv(index=False))

    checks = _claim_checks(table)
    check_pass = sum(1 for v in checks.values() if v == "pass")
    check_total = len(checks)

    summary_md = OUT_DIR / "publication_summary.md"
    with open(summary_md, "w") as f:
        f.write("# Publication Summary\n\n")
        f.write("## Main Results Table\n\n")
        f.write(f"- CSV: `{table_csv}`\n")
        f.write(f"- MD: `{table_md}`\n\n")
        f.write("## Claim Checks\n\n")
        for k, v in checks.items():
            f.write(f"- {k}: **{v}**\n")
        f.write(f"\nOverall claim checks: **{check_pass}/{check_total} passed**\n\n")
        f.write("## Notes\n\n")
        f.write("- Use only leakage-safe headline metrics (global threshold on validation, applied to test).\n")
        f.write("- Treat label-aware family thresholding as oracle diagnostics only.\n")
        f.write("- Include confidence intervals and protocol details in manuscript tables.\n")

    manifest = _environment_manifest()
    manifest["inputs"] = {
        "protocol_summary": str(PROTOCOL_SUMMARY),
        "cross_dataset_summary": str(CROSS_DATASET_SUMMARY),
    }
    manifest["outputs"] = {
        "main_metrics_table_csv": str(table_csv),
        "main_metrics_table_md": str(table_md),
        "publication_summary_md": str(summary_md),
    }
    manifest_json = OUT_DIR / "reproducibility_manifest.json"
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {table_csv}")
    print(f"Saved: {table_md}")
    print(f"Saved: {summary_md}")
    print(f"Saved: {manifest_json}")
    return OUT_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-ready artifacts from experiment outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    _parse_args()
    build_publication_bundle()

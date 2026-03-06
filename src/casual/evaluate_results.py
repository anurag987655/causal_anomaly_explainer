import os
import pickle
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.ingestion.flow_event_mapper import normalize_label


COMPONENT_COLS = ["structural_score", "intensity_score", "regularity_score", "statistical_score"]
SCORE_MODEL_CONFIGS = {
    "full_hybrid": lambda d: d["anomaly_score_0_100"] / 100.0,
    "markov_only": lambda d: d["structural_score"],
    "statistical_only": lambda d: d["statistical_score"],
    "temporal_only": lambda d: 0.5 * d["intensity_score"] + 0.5 * d["regularity_score"],
    "hybrid_without_stat": lambda d: 0.4 * d["structural_score"] + 0.4 * d["intensity_score"] + 0.2 * d["regularity_score"],
}


def _safe_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(auc(r, p))


def _f_beta(precision: float, recall: float, beta: float) -> float:
    denom = (beta ** 2) * precision + recall
    if denom <= 0:
        return 0.0
    return float((1 + beta ** 2) * precision * recall / denom)


def _resolve_beta(threshold_objective: str, threshold_beta: Optional[float]) -> Tuple[float, str]:
    obj = str(threshold_objective).strip().lower()
    if obj.startswith("f") and len(obj) > 1:
        try:
            return float(obj[1:]), f"F{float(obj[1:]):.1f}"
        except ValueError:
            pass
    if threshold_beta is not None:
        return float(threshold_beta), f"F{float(threshold_beta):.1f}"
    return 1.0, "F1.0"


def _optimize_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float) -> Tuple[float, float]:
    if len(y_true) == 0:
        return 0.5, 0.0

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0

    precision_arr = precision_arr[:-1]
    recall_arr = recall_arr[:-1]

    best_score = -1.0
    best_threshold = 0.5
    for p, r, t in zip(precision_arr, recall_arr, thresholds):
        score = _f_beta(float(p), float(r), beta)
        # Tie-break toward higher threshold (fewer false positives).
        if (score > best_score) or (np.isclose(score, best_score) and t > best_threshold):
            best_score = score
            best_threshold = float(t)

    best_threshold = float(np.clip(best_threshold, 0.0, 1.0))
    return best_threshold, float(best_score)


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}

    p_vals, r_vals, f1_vals, roc_vals, pr_vals = [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_score[idx]

        p_vals.append(precision_score(yt, yp, zero_division=0))
        r_vals.append(recall_score(yt, yp, zero_division=0))
        f1_vals.append(f1_score(yt, yp, zero_division=0))
        roc_vals.append(_safe_auc_roc(yt, ys))
        pr_vals.append(_safe_pr_auc(yt, ys))

    def ci(arr: list) -> Tuple[float, float]:
        arr = np.array(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    return {
        "precision": ci(p_vals),
        "recall": ci(r_vals),
        "f1": ci(f1_vals),
        "auc_roc": ci(roc_vals),
        "pr_auc": ci(pr_vals),
    }


def _evaluate_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": _safe_auc_roc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
    }


def _with_orientation_fix(val_y_true: np.ndarray, val_scores: np.ndarray, test_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    orientation = "higher_score_more_anomalous"
    val_auc_raw = _safe_auc_roc(val_y_true, val_scores)
    if not np.isnan(val_auc_raw) and val_auc_raw < 0.5:
        return 1.0 - val_scores, 1.0 - test_scores, "lower_score_more_anomalous (auto-inverted)"
    return val_scores, test_scores, orientation


def _ablation_table(val_df: pd.DataFrame, test_df: pd.DataFrame, beta: float) -> pd.DataFrame:
    rows = []
    for name, score_fn in SCORE_MODEL_CONFIGS.items():
        val_score = np.asarray(score_fn(val_df), dtype=float)
        test_score = np.asarray(score_fn(test_df), dtype=float)
        val_score, test_score, orientation = _with_orientation_fix(np.asarray(val_df["y_true"], dtype=int), val_score, test_score)

        threshold, val_f = _optimize_threshold(np.asarray(val_df["y_true"], dtype=int), val_score, beta)
        metrics = _evaluate_from_scores(np.asarray(test_df["y_true"], dtype=int), test_score, threshold)
        rows.append(
            {
                "model": name,
                "threshold": threshold,
                "val_fbeta": val_f,
                "orientation": orientation,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc_roc": metrics["auc_roc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    return pd.DataFrame(rows).sort_values(by="f1", ascending=False)


def _compute_model_scores(
    model_name: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_y_true: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, str, Optional[pd.DataFrame], Optional[LogisticRegression]]]:
    if model_name == "learned_hybrid":
        learned = _learn_weighted_scores(val_df, test_df)
        if learned is None:
            return None
        learned_val_scores, learned_test_scores, weights_df, learned_model = learned
        learned_val_scores, learned_test_scores, learned_orientation = _with_orientation_fix(
            val_y_true, learned_val_scores, learned_test_scores
        )
        return learned_val_scores, learned_test_scores, learned_orientation, weights_df, learned_model

    score_fn = SCORE_MODEL_CONFIGS.get(model_name)
    if score_fn is None:
        return None

    val_scores = np.asarray(score_fn(val_df), dtype=float)
    test_scores = np.asarray(score_fn(test_df), dtype=float)
    val_scores, test_scores, orientation = _with_orientation_fix(val_y_true, val_scores, test_scores)
    return val_scores, test_scores, orientation, None, None


def _learn_weighted_scores(
    val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame, LogisticRegression]]:
    if not all(c in val_df.columns for c in COMPONENT_COLS):
        return None

    X_val = val_df[COMPONENT_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_df[COMPONENT_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_val = np.asarray(val_df["y_true"], dtype=int)

    if len(np.unique(y_val)) < 2:
        return None

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_val.values, y_val)

    val_scores = model.predict_proba(X_val.values)[:, 1]
    test_scores = model.predict_proba(X_test.values)[:, 1]

    weights = pd.DataFrame(
        {
            "component": COMPONENT_COLS,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }
    )
    total_abs = weights["abs_coefficient"].sum()
    weights["normalized_abs_weight"] = (weights["abs_coefficient"] / total_abs) if total_abs > 0 else 0.0
    weights = weights.sort_values(by="abs_coefficient", ascending=False)

    intercept_row = pd.DataFrame(
        [{"component": "intercept", "coefficient": float(model.intercept_[0]), "abs_coefficient": abs(float(model.intercept_[0])), "normalized_abs_weight": np.nan}]
    )
    weights = pd.concat([weights, intercept_row], ignore_index=True)

    return val_scores, test_scores, weights, model


def _per_family_error_analysis(df_eval: pd.DataFrame, y_pred: np.ndarray, output_csv: str) -> pd.DataFrame:
    df = df_eval.copy()
    df["pred_anomaly"] = y_pred
    df["label_norm"] = df["label"].apply(normalize_label)

    families = sorted([x for x in df["label_norm"].unique() if x.upper() != "BENIGN"])
    rows = []

    for fam in families:
        y_true_f = (df["label_norm"] == fam).astype(int).values
        y_pred_a = df["pred_anomaly"].astype(int).values

        tp = int(((y_true_f == 1) & (y_pred_a == 1)).sum())
        fn = int(((y_true_f == 1) & (y_pred_a == 0)).sum())
        fp = int(((y_true_f == 0) & (y_pred_a == 1)).sum())
        tn = int(((y_true_f == 0) & (y_pred_a == 0)).sum())

        support = int(y_true_f.sum())
        recall_f = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision_f = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1_f = float((2 * precision_f * recall_f) / (precision_f + recall_f)) if (precision_f + recall_f) > 0 else 0.0

        rows.append(
            {
                "attack_family": fam,
                "support_sessions": support,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "recall": recall_f,
                "precision": precision_f,
                "f1": f1_f,
                "miss_rate": float(1.0 - recall_f),
            }
        )

    fam_df = pd.DataFrame(rows).sort_values(by=["support_sessions", "recall"], ascending=[False, True]) if rows else pd.DataFrame()
    fam_df.to_csv(output_csv, index=False)
    return fam_df


def _learn_family_thresholds(
    val_df: pd.DataFrame,
    val_scores: np.ndarray,
    default_threshold: float,
    beta: float,
    min_support: int,
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    if "label_norm" not in val_df.columns:
        return thresholds

    for family in sorted(x for x in val_df["label_norm"].unique() if x != "BENIGN"):
        fam_mask = val_df["label_norm"] == family
        benign_mask = val_df["label_norm"] == "BENIGN"
        if int(fam_mask.sum()) < min_support or int(benign_mask.sum()) < min_support:
            continue

        mask = fam_mask | benign_mask
        y_bin = np.where(fam_mask.loc[mask], 1, 0).astype(int)
        family_scores = np.asarray(val_scores[mask.values], dtype=float)
        if len(np.unique(y_bin)) < 2:
            continue

        fam_threshold, _ = _optimize_threshold(y_bin, family_scores, beta=beta)
        thresholds[family] = float(fam_threshold)
    return thresholds


def _predict_with_threshold_policy(
    scores: np.ndarray,
    labels: pd.Series,
    default_threshold: float,
    threshold_by_label: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    label_norm = labels.apply(normalize_label)
    row_thresholds = label_norm.map(lambda x: float(threshold_by_label.get(x, default_threshold))).to_numpy(dtype=float)
    y_pred = (scores >= row_thresholds).astype(int)
    return y_pred, row_thresholds


def _build_val_test_split(
    merged_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    val_fraction: float,
    random_state: int,
    split_strategy: str,
    split_min_test_anomalies: int,
    split_min_val_anomalies: int,
    split_min_test_benign: int,
    split_min_val_benign: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    strategy = str(split_strategy).strip().lower()
    if strategy == "time":
        df = merged_df.copy()
        if "session_time" not in df.columns:
            if "timestamp" in gt_df.columns and "session_id" in gt_df.columns:
                ts_map = (
                    gt_df[["session_id", "timestamp"]]
                    .assign(timestamp=pd.to_numeric(gt_df["timestamp"], errors="coerce"))
                    .dropna(subset=["timestamp"])
                    .groupby("session_id", as_index=False)["timestamp"]
                    .min()
                    .rename(columns={"timestamp": "session_time"})
                )
                df = pd.merge(df, ts_map, on="session_id", how="left")

        has_time = "session_time" in df.columns and not df["session_time"].isna().all()
        if has_time:
            df["session_time"] = pd.to_numeric(df["session_time"], errors="coerce")
            df = df.dropna(subset=["session_time"]).copy()

        # Session IDs usually end with an index from ingestion; this is a robust
        # fallback when timestamps are synthetic/relative.
        df["session_order"] = (
            df["session_id"]
            .astype(str)
            .str.extract(r"_(\d+)$", expand=False)
            .fillna("-1")
            .astype(int)
        )

        use_session_time = has_time and df["session_time"].nunique() > 10
        order_col = "session_time" if use_session_time else "session_order"
        df = df.sort_values(by=[order_col, "session_id"], ascending=True).reset_index(drop=True)

        if len(df) < 10:
            val_df, test_df = train_test_split(
                merged_df,
                test_size=val_fraction,
                random_state=random_state,
                stratify=merged_df["y_true"],
            )
            return val_df, test_df, "random (time too small fallback)"

        # Try multiple candidate split points around the target and keep the one
        # closest to requested fraction that satisfies minimum class counts.
        target_idx = int(np.floor((1.0 - val_fraction) * len(df)))
        target_idx = max(1, min(len(df) - 1, target_idx))

        candidate_fracs = np.linspace(0.55, 0.90, 15)
        candidate_idxs = sorted(
            {max(1, min(len(df) - 1, int(np.floor(fr * len(df))))) for fr in candidate_fracs} | {target_idx}
        )

        feasible: list[tuple[int, float]] = []
        y = np.asarray(df["y_true"], dtype=int)
        for idx in candidate_idxs:
            val_y = y[:idx]
            test_y = y[idx:]
            val_anom = int(val_y.sum())
            test_anom = int(test_y.sum())
            val_benign = int(len(val_y) - val_anom)
            test_benign = int(len(test_y) - test_anom)
            if (
                val_anom >= int(split_min_val_anomalies)
                and test_anom >= int(split_min_test_anomalies)
                and val_benign >= int(split_min_val_benign)
                and test_benign >= int(split_min_test_benign)
            ):
                feasible.append((idx, abs(idx - target_idx)))

        if feasible:
            split_idx = sorted(feasible, key=lambda x: x[1])[0][0]
            val_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            mode = "time" if use_session_time else "time(session_id_order_fallback)"
            return val_df, test_df, mode

        # Final fallback to random split when no non-degenerate temporal split exists.
        val_df, test_df = train_test_split(
            merged_df,
            test_size=val_fraction,
            random_state=random_state,
            stratify=merged_df["y_true"],
        )
        return val_df, test_df, "random (time non-degenerate split unavailable)"

    val_df, test_df = train_test_split(
        merged_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=merged_df["y_true"],
    )
    return val_df, test_df, "random"


def evaluate_performance(
    ground_truth_path="data/processed/all_sessions_detailed.csv",
    predictions_path="data/processed/anomaly_scores_with_features.csv",
    output_dir="data/results",
    threshold=None,
    val_fraction=0.3,
    random_state=42,
    threshold_beta=None,
    threshold_objective="f1",
    learn_hybrid_weights=True,
    n_bootstrap=400,
    learn_family_thresholds=False,
    family_threshold_min_support=50,
    split_strategy="random",
    split_min_test_anomalies=100,
    split_min_val_anomalies=100,
    split_min_test_benign=1000,
    split_min_val_benign=1000,
    score_model_preference="auto",
):
    """Research-grade evaluation with objective-tuned thresholding and optional learned hybrid weighting."""
    print("📊 Starting Research-Grade Performance Evaluation...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        beta, objective_name = _resolve_beta(threshold_objective=threshold_objective, threshold_beta=threshold_beta)

        gt_df = pd.read_csv(ground_truth_path)
        pred_df = pd.read_csv(predictions_path)

        if "label" not in pred_df.columns:
            if "label" not in gt_df.columns:
                return {"success": False, "error": "No label column found in predictions or ground truth"}
            mapping = gt_df[["session_id", "label"]].drop_duplicates()
            pred_df = pd.merge(pred_df, mapping, on="session_id", how="left")

        merged_df = pred_df.copy()
        if merged_df.empty:
            return {"success": False, "error": "No data to evaluate"}

        merged_df["label_norm"] = merged_df["label"].apply(normalize_label)
        merged_df["y_true"] = (merged_df["label_norm"] != "BENIGN").astype(int)
        merged_df["y_scores"] = pd.to_numeric(merged_df["anomaly_score_0_100"], errors="coerce").fillna(0.0) / 100.0
        merged_df = merged_df.dropna(subset=["y_scores"])

        plt.figure(figsize=(10, 6))
        sns.histplot(data=merged_df, x="anomaly_score_0_100", hue="label_norm", bins=50, kde=False)
        plt.title("Score Distribution by Attack Type")
        plt.xlabel("Anomaly Score (0-100)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_distribution.png")
        plt.close()

        selected_model_name = "full_hybrid"
        score_orientation = "higher_score_more_anomalous"
        policy_path = None
        deployed_meta_model = None
        deployed_feature_cols = COMPONENT_COLS.copy()
        deployed_calibrator_model = None
        deployed_family_thresholds = {}

        if threshold is None:
            val_df, test_df, split_mode_used = _build_val_test_split(
                merged_df=merged_df,
                gt_df=gt_df,
                val_fraction=val_fraction,
                random_state=random_state,
                split_strategy=split_strategy,
                split_min_test_anomalies=int(split_min_test_anomalies),
                split_min_val_anomalies=int(split_min_val_anomalies),
                split_min_test_benign=int(split_min_test_benign),
                split_min_val_benign=int(split_min_val_benign),
            )

            val_y_true = np.asarray(val_df["y_true"], dtype=int)
            test_y_true = np.asarray(test_df["y_true"], dtype=int)

            weights_csv = None
            preference = str(score_model_preference or "auto").strip().lower()
            supported = {"auto", "learned_hybrid"} | set(SCORE_MODEL_CONFIGS.keys())

            if preference not in supported:
                print(f"⚠️ Unknown score_model_preference='{score_model_preference}', falling back to 'auto'")
                preference = "auto"

            if preference == "auto":
                candidates = ["full_hybrid"]
                if learn_hybrid_weights:
                    candidates.append("learned_hybrid")
            else:
                candidates = [preference]

            best_val_scores = None
            best_test_scores = None
            best_threshold = 0.5
            best_f_beta = -1.0

            for candidate in candidates:
                scored = _compute_model_scores(candidate, val_df, test_df, val_y_true)
                if scored is None:
                    print(f"⚠️ Skipping unavailable score model: {candidate}")
                    continue

                cand_val_scores, cand_test_scores, cand_orientation, cand_weights_df, cand_model = scored
                cand_threshold, cand_val_f = _optimize_threshold(val_y_true, cand_val_scores, beta)

                if (cand_val_f > best_f_beta) or (
                    np.isclose(cand_val_f, best_f_beta) and cand_threshold > best_threshold
                ):
                    selected_model_name = candidate
                    best_val_scores = cand_val_scores
                    best_test_scores = cand_test_scores
                    best_threshold = cand_threshold
                    best_f_beta = cand_val_f
                    score_orientation = cand_orientation
                    deployed_meta_model = cand_model
                    deployed_feature_cols = COMPONENT_COLS.copy()

                if cand_weights_df is not None:
                    weights_csv = f"{output_dir}/learned_hybrid_weights.csv"
                    cand_weights_df.to_csv(weights_csv, index=False)

            if best_val_scores is None or best_test_scores is None:
                return {"success": False, "error": "No valid scoring model could be evaluated"}

            # Primary research metrics must be leakage-safe:
            # use one global threshold learned on validation and applied on test.
            y_pred = (best_test_scores >= float(best_threshold)).astype(int)
            test_row_thresholds = np.full(shape=len(best_test_scores), fill_value=float(best_threshold), dtype=float)
            oracle_metrics = None
            oracle_thresholds_csv = ""

            if learn_family_thresholds:
                deployed_family_thresholds = _learn_family_thresholds(
                    val_df=val_df,
                    val_scores=best_val_scores,
                    default_threshold=best_threshold,
                    beta=beta,
                    min_support=int(family_threshold_min_support),
                )

                # Optional oracle-only analysis (not headline metrics).
                y_pred_oracle, _ = _predict_with_threshold_policy(
                    scores=best_test_scores,
                    labels=test_df["label_norm"],
                    default_threshold=best_threshold,
                    threshold_by_label=deployed_family_thresholds,
                )
                oracle_precision = float(precision_score(test_y_true, y_pred_oracle, zero_division=0))
                oracle_recall = float(recall_score(test_y_true, y_pred_oracle, zero_division=0))
                oracle_f1 = float(f1_score(test_y_true, y_pred_oracle, zero_division=0))
                oracle_metrics = {
                    "precision": oracle_precision,
                    "recall": oracle_recall,
                    "f1": oracle_f1,
                    "family_threshold_count": int(len(deployed_family_thresholds)),
                }
                oracle_thresholds_csv = f"{output_dir}/family_thresholds_oracle.csv"
                pd.DataFrame(
                    [{"label": k, "threshold": v} for k, v in sorted(deployed_family_thresholds.items())]
                ).to_csv(oracle_thresholds_csv, index=False)

            split_desc = f"{split_mode_used} | validation={len(val_df)}, test={len(test_df)}"
            eval_size = len(test_df)
            eval_anom = int(test_y_true.sum())

            calibrated_probs = best_test_scores.copy()
            brier = float("nan")
            ece = float("nan")
            if len(np.unique(val_y_true)) >= 2:
                cal_model = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=2000)
                cal_model.fit(best_val_scores.reshape(-1, 1), val_y_true)
                calibrated_probs = cal_model.predict_proba(best_test_scores.reshape(-1, 1))[:, 1]
                deployed_calibrator_model = cal_model
                brier = float(brier_score_loss(test_y_true, calibrated_probs))

                frac_pos, mean_pred = calibration_curve(test_y_true, calibrated_probs, n_bins=10, strategy="quantile")
                if len(frac_pos) > 0:
                    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

                plt.figure(figsize=(6, 6))
                plt.plot([0, 1], [0, 1], "k--", label="Ideal")
                plt.plot(mean_pred, frac_pos, marker="o", label="Calibrated")
                plt.xlabel("Predicted probability")
                plt.ylabel("Observed anomaly frequency")
                plt.title("Calibration Curve (Test)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/calibration_curve.png")
                plt.close()

            precision = float(precision_score(test_y_true, y_pred, zero_division=0))
            recall = float(recall_score(test_y_true, y_pred, zero_division=0))
            test_f1 = float(f1_score(test_y_true, y_pred, zero_division=0))
            auc_roc = _safe_auc_roc(test_y_true, best_test_scores)
            pr_auc = _safe_pr_auc(test_y_true, best_test_scores)
            calibrated_pr_auc = _safe_pr_auc(test_y_true, calibrated_probs)

            ci = _bootstrap_ci(test_y_true, y_pred, best_test_scores, n_bootstrap=n_bootstrap, seed=random_state)
            ablation_df = _ablation_table(val_df, test_df, beta=beta)
            ablation_csv = f"{output_dir}/ablation_results.csv"
            ablation_df.to_csv(ablation_csv, index=False)

            pr_p, pr_r, _ = precision_recall_curve(test_y_true, best_test_scores)
            plt.figure(figsize=(8, 6))
            plt.plot(pr_r, pr_p, label=f"PR curve (AUPRC={pr_auc:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (Test, {selected_model_name})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/precision_recall_curve.png")
            plt.close()

            eval_labels_df = test_df.copy()
            eval_labels_df = eval_labels_df.assign(eval_score=best_test_scores)
            eval_labels_df = eval_labels_df.assign(applied_threshold=test_row_thresholds)

            # Save deployment-ready policy artifact for inference-time scoring.
            policy = {
                "version": 1,
                "score_model": selected_model_name,
                "score_orientation": score_orientation,
                "threshold": float(best_threshold),
                "threshold_objective": objective_name,
                "feature_cols": deployed_feature_cols,
                "meta_model": deployed_meta_model,
                "calibrator_model": deployed_calibrator_model,
                "threshold_by_label": deployed_family_thresholds,
            }
            policy_path = f"{output_dir}/deployment_policy.pkl"
            with open(policy_path, "wb") as f:
                pickle.dump(policy, f)

            family_thresholds_csv = oracle_thresholds_csv
        else:
            best_threshold = float(threshold)
            y_true = np.asarray(merged_df["y_true"], dtype=int)
            y_scores = np.asarray(merged_df["y_scores"], dtype=float)
            y_pred, row_thresholds = _predict_with_threshold_policy(
                scores=y_scores,
                labels=merged_df["label_norm"],
                default_threshold=best_threshold,
                threshold_by_label={},
            )
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            test_f1 = float(f1_score(y_true, y_pred, zero_division=0))
            auc_roc = _safe_auc_roc(y_true, y_scores)
            pr_auc = _safe_pr_auc(y_true, y_scores)
            calibrated_pr_auc = float("nan")
            brier = float("nan")
            ece = float("nan")
            ci = _bootstrap_ci(y_true, y_pred, y_scores, n_bootstrap=n_bootstrap, seed=random_state)
            split_desc = "full dataset (fixed threshold)"
            eval_size = len(merged_df)
            eval_anom = int(y_true.sum())
            best_f_beta = float("nan")
            ablation_csv = ""
            weights_csv = None
            policy_path = None
            family_thresholds_csv = ""
            oracle_metrics = None
            split_mode_used = "fixed_threshold"
            eval_labels_df = merged_df.copy()
            eval_labels_df = eval_labels_df.assign(eval_score=y_scores)
            eval_labels_df = eval_labels_df.assign(applied_threshold=row_thresholds)

        per_label = (
            eval_labels_df.groupby("label")
            .agg(sessions=("session_id", "count"), anomaly_rate=("y_true", "mean"), mean_score=("eval_score", "mean"))
            .sort_values(by="mean_score", ascending=False)
        )
        per_label_csv = f"{output_dir}/per_label_analysis.csv"
        per_label.to_csv(per_label_csv)

        family_errors_csv = f"{output_dir}/per_family_error_analysis.csv"
        fam_err_df = _per_family_error_analysis(eval_labels_df, y_pred=y_pred, output_csv=family_errors_csv)

        results_txt = (
            f"--- Performance Report (Threshold: {best_threshold:.4f}) ---\n"
            f"Evaluation set: {split_desc}\n"
            f"Threshold objective: {objective_name} on validation\n"
            f"Split strategy: {split_mode_used}\n"
            f"Validation best {objective_name}: {best_f_beta:.4f}\n"
            f"Selected scoring model: {selected_model_name}\n"
            f"Score orientation: {score_orientation}\n"
            f"Family-threshold oracle learned: {len(deployed_family_thresholds)}\n"
            f"Total Sessions: {eval_size}\n"
            f"Anomalies (GT): {eval_anom} | Benign (GT): {eval_size - eval_anom}\n"
            f"--------------------------\n"
            f"Precision:   {precision:.4f} (95% CI {ci['precision'][0]:.4f}-{ci['precision'][1]:.4f})\n"
            f"Recall:      {recall:.4f} (95% CI {ci['recall'][0]:.4f}-{ci['recall'][1]:.4f})\n"
            f"F1-Score:    {test_f1:.4f} (95% CI {ci['f1'][0]:.4f}-{ci['f1'][1]:.4f})\n"
            f"AUC-ROC:     {auc_roc:.4f} (95% CI {ci['auc_roc'][0]:.4f}-{ci['auc_roc'][1]:.4f})\n"
            f"PR-AUC:      {pr_auc:.4f} (95% CI {ci['pr_auc'][0]:.4f}-{ci['pr_auc'][1]:.4f})\n"
            f"Cal PR-AUC:  {calibrated_pr_auc:.4f}\n"
            f"Brier score: {brier:.4f}\n"
            f"ECE (10-bin):{ece:.4f}\n"
            f"--------------------------\n"
            f"Per-label analysis: {per_label_csv}\n"
            f"Per-family errors: {family_errors_csv} ({len(fam_err_df)} families)\n"
            f"Ablations: {ablation_csv or 'n/a (fixed threshold mode)'}\n"
            f"Learned weights: {weights_csv or 'n/a'}\n"
            f"Family-threshold oracle table: {family_thresholds_csv or 'n/a'}\n"
            f"Oracle metrics (label-aware thresholding): {oracle_metrics or 'disabled'}\n"
            f"Deployment policy: {policy_path or 'n/a (fixed threshold mode)'}\n"
        )
        print(results_txt)

        report_path = f"{output_dir}/evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write(results_txt)

        cm_source_y = np.asarray(eval_labels_df["y_true"], dtype=int)
        cm = confusion_matrix(cm_source_y, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Benign", "Anomaly"], yticklabels=["Benign", "Anomaly"])
        plt.title(f"Confusion Matrix (Threshold: {best_threshold:.2f})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_optimal.png")
        plt.close()

        return {
            "success": True,
            "threshold": float(best_threshold),
            "threshold_objective": objective_name,
            "split_strategy": split_mode_used,
            "score_model": selected_model_name,
            "score_orientation": score_orientation,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(test_f1),
            "auc_roc": float(auc_roc),
            "pr_auc": float(pr_auc),
            "calibrated_pr_auc": float(calibrated_pr_auc) if not np.isnan(calibrated_pr_auc) else None,
            "brier_score": float(brier) if not np.isnan(brier) else None,
            "ece": float(ece) if not np.isnan(ece) else None,
            "evaluation_sessions": int(eval_size),
            "evaluation_anomalies": int(eval_anom),
            "report_path": report_path,
            "per_label_csv": per_label_csv,
            "per_family_error_csv": family_errors_csv,
            "ablation_csv": ablation_csv or None,
            "learned_weights_csv": weights_csv,
            "family_thresholds_csv": family_thresholds_csv or None,
            "oracle_metrics": oracle_metrics,
            "deployment_policy_path": policy_path,
        }

    except Exception as e:
        print(f"❌ Eval Error: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    evaluate_performance()

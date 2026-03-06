"""
Law Enforcement Scorer - Weighted Multi-Factor Mode
Combines structural Markov probability, event intensity, and timing regularity.
"""
import pickle
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.ingestion.flow_event_mapper import normalize_label

class MarkovAnomalyScorer:
    def __init__(self, 
                 dag_model_path: str = "data/processed/dag_model_complete.pkl",
                 train_features_path: str = "data/processed/baseline_features.csv",
                 anomaly_percentile: float = 5.0,
                 deployment_policy_path: Optional[str] = None):
        self.min_prob = 1e-6
        self.anomaly_percentile = anomaly_percentile
        self.hybrid_feature_cols = [
            'event_count',
            'duration_seconds',
            'bytes_transferred',
            'avg_event_rate',
            'repetition_rate',
            'unique_events_count',
            'entropy_of_transition_probs',
            'downstream_diversity_ports',
            'ratio_large_transfer_events',
            'burstiness',
            'dst_port',
            'protocol',
        ]
        self.scaler = None
        self.iso_forest = None
        self.deployment_policy = self._load_deployment_policy(deployment_policy_path)
        
        with open(dag_model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.transition_probs = model_data['probabilities']
        self._fit_statistical_model(train_features_path)
        print(f"✅ Law Enforcement Scorer Active (Weighted Multi-Factor Mode)")
        if self.deployment_policy is not None:
            print("✅ Deployment policy loaded (meta calibration + threshold policy)")

    def _load_deployment_policy(self, deployment_policy_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not deployment_policy_path:
            return None
        if not os.path.exists(deployment_policy_path):
            print(f"⚠️ Deployment policy not found: {deployment_policy_path} (falling back to default scoring)")
            return None
        try:
            with open(deployment_policy_path, "rb") as f:
                policy = pickle.load(f)
            if not isinstance(policy, dict):
                print("⚠️ Deployment policy format invalid (expected dict), falling back to default scoring")
                return None
            return policy
        except Exception as e:
            print(f"⚠️ Failed to load deployment policy: {e} (falling back to default scoring)")
            return None
    
    def _fit_statistical_model(self, train_features_path: str):
        """Fit IsolationForest on benign baseline feature vectors."""
        if not train_features_path or not os.path.exists(train_features_path):
            return
        
        try:
            train_df = pd.read_csv(train_features_path)
            available = [c for c in self.hybrid_feature_cols if c in train_df.columns]
            if len(available) < 5:
                return
            
            X = train_df[available].replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
            self.hybrid_feature_cols = available
            self.scaler = RobustScaler(quantile_range=(5, 95))
            Xs = self.scaler.fit_transform(X)
            self.iso_forest = IsolationForest(
                n_estimators=200,
                contamination=min(0.2, max(0.01, self.anomaly_percentile / 100.0)),
                random_state=42,
                n_jobs=1,
            )
            self.iso_forest.fit(Xs)
        except Exception:
            self.scaler = None
            self.iso_forest = None

    def _get_prob(self, src, dst):
        return self.transition_probs.get((src, dst), self.min_prob)
    
    def _session_feature_vector(self, group: pd.DataFrame, events: list, transition_probs: list) -> dict:
        duration = float(max(group['timestamp']) - min(group['timestamp']))
        if duration < 0:
            duration = 0.0
        
        bytes_total = 0.0
        if 'session_bytes_total' in group.columns and pd.notna(group['session_bytes_total'].iloc[0]):
            bytes_total = float(group['session_bytes_total'].iloc[0])
        elif 'length' in group.columns:
            bytes_total = float(pd.to_numeric(group['length'], errors='coerce').fillna(0.0).sum())
        
        avg_event_rate = (len(events) / duration) if duration > 0 else float(len(events))
        unique_events = len(set(events))
        repetition_rate = 1 - (unique_events / len(events)) if len(events) > 0 else 0.0
        
        probs_array = np.array(transition_probs, dtype=float)
        probs_array = probs_array[probs_array > 0]
        entropy = float(-np.sum(probs_array * np.log(probs_array))) if len(probs_array) > 0 else 0.0
        
        downstream_ports = (
            group['dst_port'].nunique()
            if 'dst_port' in group.columns
            else (group['session_unique_dst_ports'].iloc[0] if 'session_unique_dst_ports' in group.columns else 1)
        )
        large_count = sum(1 for e in events if e == 'LARGE_TRANSFER')
        ratio_large = (large_count / len(events)) if len(events) > 0 else 0.0
        burstiness = (
            float(group['session_max_packets_per_sec'].iloc[0])
            if 'session_max_packets_per_sec' in group.columns and pd.notna(group['session_max_packets_per_sec'].iloc[0])
            else max(1.0, avg_event_rate)
        )
        dst_port = (
            float(group['dst_port'].iloc[0])
            if 'dst_port' in group.columns and pd.notna(group['dst_port'].iloc[0])
            else 0.0
        )
        protocol = (
            float(group['protocol'].iloc[0])
            if 'protocol' in group.columns and pd.notna(group['protocol'].iloc[0])
            else 0.0
        )
        
        return {
            'event_count': float(len(events)),
            'duration_seconds': duration,
            'bytes_transferred': bytes_total,
            'avg_event_rate': avg_event_rate,
            'repetition_rate': repetition_rate,
            'unique_events_count': float(unique_events),
            'entropy_of_transition_probs': entropy,
            'downstream_diversity_ports': float(downstream_ports),
            'ratio_large_transfer_events': ratio_large,
            'burstiness': float(burstiness),
            'dst_port': dst_port,
            'protocol': protocol,
        }

    def calculate_session_metrics(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        print("🔍 Calculating Weighted Multi-Factor Scores (Structural 0.3, Intensity 0.5, Regularity 0.2)...")
        rows = []
        feature_rows = []
        
        # Ensure timestamp is numeric
        sessions_df['timestamp'] = pd.to_numeric(sessions_df['timestamp'], errors='coerce')
        sessions_df = sessions_df.dropna(subset=['timestamp'])

        # Infer timestamp unit once to avoid ms/seconds mismatch.
        # Epoch timestamps are usually > 1e9 (seconds) or > 1e12 (milliseconds).
        ts_median = sessions_df['timestamp'].median()
        timestamps_are_ms = ts_median > 1e11
        
        for session_id, group in sessions_df.groupby('session_id'):
            # Sort by timestamp to ensure sequence and IATs are correct
            group = group.sort_values('timestamp')
            events = group['event'].tolist()
            timestamps = group['timestamp'].tolist()
            
            if len(events) < 2: 
                continue
            
            # 1. Structural Score: Markov Log-Probability
            # Higher score for unseen/rare transitions (-log(prob))
            neg_log_probs = []
            trans_probs = []
            for i in range(len(events) - 1):
                prob = self._get_prob(events[i], events[i+1])
                trans_probs.append(prob)
                neg_log_probs.append(-np.log(max(prob, self.min_prob)))
            
            avg_neg_log = np.mean(neg_log_probs)
            # Normalize: -log(1e-6) approx 13.8. Cap at 15 for 0-1 scale.
            structural_norm = min(1.0, avg_neg_log / 15.0)
            
            # 2. Intensity Score: Event Rate (Events per second)
            # High-speed sessions should be heavily penalized.
            duration_raw = max(timestamps) - min(timestamps)
            duration_sec = duration_raw / 1000.0 if timestamps_are_ms else duration_raw
            
            if duration_sec <= 0:
                # Instant sessions with multiple events are extremely high intensity
                event_rate = len(events) * 5.0 
            else:
                event_rate = len(events) / duration_sec
            
            # Normalize: 50 events/sec is high for many network activities.
            intensity_norm = min(1.0, event_rate / 50.0)
            
            # 3. Regularity Score: IAT Variance
            # If timing is 'too perfect' (low variance), it's a script.
            iats = np.diff(timestamps)
            if len(iats) > 1:
                iat_var = np.var(iats)
                # Map low variance to high score. 
                # If variance is 0, score is 1.0. As variance increases, score drops.
                # Scaling factor 500.0 means a std_dev of ~22ms starts reducing score significantly.
                regularity_norm = np.exp(-iat_var / 500.0)
            else:
                # Neutral for very short sessions
                regularity_norm = 0.5
            
            session_feats = self._session_feature_vector(group, events, trans_probs)
            
            # Determine session label (Prioritize attack labels over BENIGN)
            all_labels = group['label'].unique()
            non_benign_labels = [l for l in all_labels if normalize_label(l) != 'BENIGN']
            if non_benign_labels:
                session_label = normalize_label(non_benign_labels[0])
            else:
                session_label = 'BENIGN'
            
            rows.append({
                'session_id': session_id,
                'structural_score': structural_norm,
                'intensity_score': intensity_norm,
                'regularity_score': regularity_norm,
                'label': session_label
            })
            feature_rows.append(session_feats)
            
            # Per-session attack logging is intentionally disabled to keep large experiment runs readable.
        
        scores_df = pd.DataFrame(rows)
        if scores_df.empty:
            return scores_df
        
        # 4. Statistical Score: IsolationForest on baseline-trained feature vectors (batched)
        if self.iso_forest is not None and self.scaler is not None:
            feat_df = pd.DataFrame(feature_rows)
            X = feat_df.reindex(columns=self.hybrid_feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            Xs = self.scaler.transform(X)
            normality = self.iso_forest.decision_function(Xs)
            stat_scores = np.clip(0.5 - normality, 0.0, 1.0)
            scores_df['statistical_score'] = stat_scores
        else:
            scores_df['statistical_score'] = 0.5

        # Hybrid combine: Markov + statistical detector baseline.
        base_score = (
            (scores_df['structural_score'] * 0.30) +
            (scores_df['intensity_score'] * 0.30) +
            (scores_df['regularity_score'] * 0.15) +
            (scores_df['statistical_score'] * 0.25)
        )
        scores_df['anomaly_score_0_100'] = 100.0 * base_score

        # Optional deployment policy: learned fusion + orientation fix + calibration.
        if self.deployment_policy is not None:
            raw_score = np.asarray(base_score, dtype=float)
            meta_model = self.deployment_policy.get("meta_model")
            feature_cols = self.deployment_policy.get(
                "feature_cols",
                ["structural_score", "intensity_score", "regularity_score", "statistical_score"],
            )
            if meta_model is not None:
                X_meta = (
                    scores_df.reindex(columns=feature_cols)
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                raw_score = meta_model.predict_proba(X_meta.values)[:, 1]

            orientation = str(self.deployment_policy.get("score_orientation", "higher_score_more_anomalous")).lower()
            if orientation.startswith("lower_score_more_anomalous"):
                raw_score = 1.0 - raw_score

            calibrated_prob = raw_score.copy()
            calibrator_model = self.deployment_policy.get("calibrator_model")
            if calibrator_model is not None:
                calibrated_prob = calibrator_model.predict_proba(raw_score.reshape(-1, 1))[:, 1]

            scores_df["decision_score"] = np.clip(raw_score, 0.0, 1.0)
            scores_df["calibrated_anomaly_prob"] = np.clip(calibrated_prob, 0.0, 1.0)
            scores_df['anomaly_score_0_100'] = 100.0 * scores_df["calibrated_anomaly_prob"]

        return scores_df

    def detect_anomalies(self, scores_df: pd.DataFrame):
        # If deployment policy contains threshold policy, use it in score space [0,1].
        policy_threshold = None
        if self.deployment_policy is not None and self.deployment_policy.get("threshold") is not None:
            policy_threshold = float(self.deployment_policy["threshold"])
        label_thresholds = {}
        if self.deployment_policy is not None and isinstance(self.deployment_policy.get("threshold_by_label"), dict):
            label_thresholds = {
                str(k): float(v)
                for k, v in self.deployment_policy["threshold_by_label"].items()
            }

        score_space = (
            pd.to_numeric(scores_df["decision_score"], errors="coerce").fillna(0.0)
            if "decision_score" in scores_df.columns
            else pd.to_numeric(scores_df["anomaly_score_0_100"], errors="coerce").fillna(0.0) / 100.0
        )

        if label_thresholds and "label" in scores_df.columns:
            default_threshold = (
                float(np.clip(policy_threshold, 0.0, 1.0))
                if policy_threshold is not None
                else float(scores_df["anomaly_score_0_100"].quantile(1.0 - (self.anomaly_percentile / 100.0)) / 100.0)
            )
            label_norm = scores_df["label"].apply(normalize_label)
            row_thresholds = label_norm.map(lambda x: float(np.clip(label_thresholds.get(x, default_threshold), 0.0, 1.0)))
            scores_df["is_anomaly"] = score_space >= row_thresholds
            scores_df["anomaly_threshold"] = row_thresholds
            threshold = default_threshold
        elif policy_threshold is not None:
            threshold = float(np.clip(policy_threshold, 0.0, 1.0))
            scores_df['is_anomaly'] = score_space >= threshold
            scores_df['anomaly_threshold'] = threshold
        else:
            # Mark top anomaly_percentile sessions as anomalous.
            q = 1.0 - (self.anomaly_percentile / 100.0)
            threshold = float(scores_df['anomaly_score_0_100'].quantile(q))
            scores_df['is_anomaly'] = scores_df['anomaly_score_0_100'] >= threshold
            scores_df['anomaly_threshold'] = threshold
        return scores_df.sort_values('anomaly_score_0_100', ascending=False)

    def explain_top_anomalies(self, *args, **kwargs): return pd.DataFrame()
    def save_rca_results(self, *args, **kwargs): pass

def run_anomaly_scoring_with_rca(**kwargs):
    # Honor caller-provided paths to keep experiments reproducible.
    data_path = kwargs.get("sessions_csv", "data/processed/all_sessions_detailed.csv")
    model_path = kwargs.get("dag_model", "data/processed/dag_model_complete.pkl")
    output_path = kwargs.get("output_csv", "data/processed/anomaly_scores_with_features.csv")
    train_features = kwargs.get("train_features", "data/processed/baseline_features.csv")
    anomaly_percentile = float(kwargs.get("anomaly_percentile", 5.0))
    deployment_policy = kwargs.get("deployment_policy")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return {"success": False, "error": "Model file not found"}

    scorer = MarkovAnomalyScorer(
        dag_model_path=model_path,
        train_features_path=train_features,
        anomaly_percentile=anomaly_percentile,
        deployment_policy_path=deployment_policy,
    )
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return {"success": False, "error": "Data file not found"}
        
    sessions_df = pd.read_csv(data_path)
    scores_df = scorer.calculate_session_metrics(sessions_df)
    anomalies_df = scorer.detect_anomalies(scores_df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anomalies_df.to_csv(output_path, index=False)
    print(f"✅ Anomaly scores saved to {output_path}")
    return {"success": True}

if __name__ == "__main__":
    run_anomaly_scoring_with_rca()

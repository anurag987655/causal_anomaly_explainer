"""
Law Engine Baseline: Learns the 'Natural Laws' of the Monday network.
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ingestion.flow_event_mapper import infer_flow_columns, extract_flow_features, build_session_rows


def prepare_baseline(
    raw_path="data/raw/Monday-WorkingHours.pcap_ISCX.csv",
    output_path="data/processed/baseline_training.csv",
    max_rows=None,
):
    print("🚀 Teaching the Law Engine 'Normal Behavior'...")
    df = pd.read_csv(raw_path)
    if max_rows is not None and int(max_rows) > 0:
        df = df.head(int(max_rows))
    df.columns = [c.strip() for c in df.columns]
    colmap = infer_flow_columns(df)

    sessions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Learning Rules"):
        flow = extract_flow_features(row, colmap, fallback_label="BENIGN")
        session_id = (
            f"BENIGN_{flow['src_ip']}_{flow['dst_ip']}_"
            f"{flow['src_port']}_{flow['dst_port']}_{flow['protocol']}_{idx}"
        )
        sessions.extend(build_session_rows(session_id, flow))
            
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sessions).to_csv(output_path, index=False)
    print(f"✅ Normal Laws learned.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare baseline training sessions from flow CSV")
    parser.add_argument("--raw-path", default="data/raw/Monday-WorkingHours.pcap_ISCX.csv")
    parser.add_argument("--output-path", default="data/processed/baseline_training.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick experiments")
    args = parser.parse_args()
    prepare_baseline(raw_path=args.raw_path, output_path=args.output_path, max_rows=args.max_rows)

"""
NetCausalAI - Behavioral Law Engine
Learns the 'Laws of Dependency' between different network services.
"""
import argparse
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

from src.ingestion.flow_event_mapper import infer_flow_columns, extract_flow_features, build_session_rows


def _parse_dataset_spec(spec: str):
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset spec '{spec}'. Expected format: filename.csv:LabelHint")
    return parts[0].strip(), parts[1].strip()


def load_and_transform_combined(
    raw_dir="data/raw",
    output_path="data/processed/all_sessions_detailed.csv",
    files=None,
    max_rows_per_file=None,
):
    print("🚀 [LAW ENGINE] Learning Behavioral Dependencies...")

    files = files or [
        ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "PortScan"),
        ("Tuesday-WorkingHours.pcap_ISCX.csv", "BENIGN"),
    ]

    all_events = []

    for filename, type_hint in files:
        path = os.path.join(raw_dir, filename)
        if not os.path.exists(path):
            continue

        print(f"📂 Analyzing Laws in {filename}...")
        # Build flow-level sessions to avoid mixing unrelated traffic.
        df = pd.read_csv(path)
        if max_rows_per_file is not None and int(max_rows_per_file) > 0:
            df = df.head(int(max_rows_per_file))
        df.columns = [c.strip() for c in df.columns]
        colmap = infer_flow_columns(df)
        file_tag = Path(filename).stem

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Discovering Behaviors"):
            flow = extract_flow_features(row, colmap, fallback_label=type_hint)
            session_id = (
                f"{file_tag}_{flow['src_ip']}_{flow['dst_ip']}_"
                f"{flow['src_port']}_{flow['dst_port']}_{flow['protocol']}_{idx}"
            )
            all_events.extend(build_session_rows(session_id, flow))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_events).to_csv(output_path, index=False)
    print(f"✅ Behavioral sequences ready for Law Enforcement.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build combined evaluation sessions from one or more flow CSV files")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-path", default="data/processed/all_sessions_detailed.csv")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec in format filename.csv:LabelHint (repeatable)",
    )
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per dataset")
    args = parser.parse_args()

    file_specs = [_parse_dataset_spec(spec) for spec in args.dataset] if args.dataset else None
    load_and_transform_combined(
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        files=file_specs,
        max_rows_per_file=args.max_rows_per_file,
    )

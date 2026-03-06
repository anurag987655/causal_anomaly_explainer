import pandas as pd
import numpy as np
import os
from tqdm import tqdm # Added for a progress bar

# Configuration
RAW_DATA = "/home/anurag/Projects/NetCausalAI_test/data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
OUTPUT_FILE = "data/processed/all_sessions_detailed.csv"
SAMPLE_SIZE = 20000  # <--- Change this to 30,000 for lightning speed

def transform_kaggle_fast():
    print(f"🚀 [IDX FAST MODE] Reading {SAMPLE_SIZE} flows from {RAW_DATA}...")
    
    # Load limited rows
    try:
        df = pd.read_csv(RAW_DATA, nrows=SAMPLE_SIZE, skiprows=range(1, 100000))
    except FileNotFoundError:
        print("❌ Error: Could not find the Kaggle CSV. Check your data/raw/ folder.")
        return

    df.columns = df.columns.str.strip()
    event_rows = []
    
    # tqdm adds a progress bar so you can see the speed in IDX
    print("Mapping flows to sequences...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sess_id = f"k_{idx}"
        
        # 1. Start
        start_evt = "TCP_SYN" if row['SYN Flag Count'] > 0 else "UDP_PACKET"
        # 2. Middle
        mid_evt = "LARGE_TRANSFER" if row['Total Length of Fwd Packets'] > 1000 else "SMALL_TRANSFER"
        
        base = [
            row['Flow Duration'], 
            row['Total Length of Fwd Packets'] + row['Total Length of Bwd Packets'],
            row['Flow Packets/s'],
            row['Destination Port'],
            row['Label']
        ]

        event_rows.append([sess_id, 0.0, start_evt, 0] + base)
        event_rows.append([sess_id, row['Flow Duration']/2, mid_evt, row['Total Length of Fwd Packets']] + base)
        event_rows.append([sess_id, row['Flow Duration'], "SESSION_END", 0] + base)

    cols = ["session_id", "timestamp", "event", "length", 
            "session_duration", "session_bytes_total", "session_packet_rate", 
            "dst_port", "label"]
    
    final_df = pd.DataFrame(event_rows, columns=cols)
    os.makedirs("data/processed", exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Success! Created {len(final_df)} events in {OUTPUT_FILE}")

if __name__ == "__main__":
    transform_kaggle_fast()
"""
Shared flow -> semantic event mapping for NetCausalAI.

This keeps baseline training, mixed-data generation, and experiment datasets
on the same event vocabulary and sequence construction rules.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


LABEL_ALIASES = {
    "BENIGN": "BENIGN",
    "PORTSCAN": "PortScan",
    "DDOS": "DDoS",
    "DOS HULK": "DoS Hulk",
    "DOS SLOWLORIS": "DoS slowloris",
    "DOS SLOWHTTPTEST": "DoS Slowhttptest",
    "BOT": "Bot",
    "INFILTRATION": "Infiltration",
    "FTP PATATOR": "FTP-Patator",
    "SSH PATATOR": "SSH-Patator",
    "WEB ATTACK BRUTE FORCE": "Web Attack - Brute Force",
    "WEB ATTACK XSS": "Web Attack - XSS",
    "WEB ATTACK SQL INJECTION": "Web Attack - Sql Injection",
}


def normalize_label(label: object, fallback: str = "BENIGN") -> str:
    if label is None:
        return fallback

    text = str(label).strip()
    if not text:
        return fallback

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().upper()
    if not text:
        return fallback

    return LABEL_ALIASES.get(text, text.title())


def pick_col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return default


def infer_flow_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "src_ip": pick_col(df, ["Source IP", "src_ip"], None),
        "dst_ip": pick_col(df, ["Destination IP", "dst_ip"], None),
        "src_port": pick_col(df, ["Source Port", "src_port"], None),
        "dst_port": pick_col(df, ["Destination Port", "dst_port"], None),
        "proto": pick_col(df, ["Protocol", "protocol"], None),
        "duration": pick_col(df, ["Flow Duration", "flow_duration", "session_duration"], None),
        "packet_rate": pick_col(df, ["Flow Packets/s", "flow packets/s", "session_packet_rate"], None),
        "fwd_len": pick_col(df, ["Total Length of Fwd Packets"], None),
        "bwd_len": pick_col(df, ["Total Length of Bwd Packets"], None),
        "fwd_pkts": pick_col(df, ["Total Fwd Packets", "total_fwd_packets"], None),
        "bwd_pkts": pick_col(df, ["Total Backward Packets", "total_bwd_packets"], None),
        "flow_iat_mean": pick_col(df, ["Flow IAT Mean", "flow_iat_mean"], None),
        "flow_iat_std": pick_col(df, ["Flow IAT Std", "flow_iat_std"], None),
        "pkt_len_mean": pick_col(df, ["Packet Length Mean", "packet_length_mean"], None),
        "pkt_len_std": pick_col(df, ["Packet Length Std", "packet_length_std"], None),
        "syn_count": pick_col(df, ["SYN Flag Count", "syn flag count"], None),
        "label": pick_col(df, ["Label", "label"], None),
    }


def _safe_int(row: pd.Series, col: Optional[str], default: int = 0) -> int:
    if col is None or col not in row.index or pd.isna(row[col]):
        return default
    return int(row[col])


def _safe_float(row: pd.Series, col: Optional[str], default: float = 0.0) -> float:
    if col is None or col not in row.index or pd.isna(row[col]):
        return default
    value = float(row[col])
    if not np.isfinite(value):
        return default
    return value


def extract_flow_features(row: pd.Series, colmap: Dict[str, Optional[str]], fallback_label: str = "BENIGN") -> Dict:
    src_ip = row[colmap["src_ip"]] if colmap["src_ip"] and pd.notna(row[colmap["src_ip"]]) else "unknown_src"
    dst_ip = row[colmap["dst_ip"]] if colmap["dst_ip"] and pd.notna(row[colmap["dst_ip"]]) else "unknown_dst"
    src_port = _safe_int(row, colmap["src_port"], 0)
    dst_port = _safe_int(row, colmap["dst_port"], 0)
    proto = _safe_int(row, colmap["proto"], 0)
    duration = _safe_float(row, colmap["duration"], 1.0)
    if duration <= 0:
        duration = 1.0
    packet_rate = _safe_float(row, colmap["packet_rate"], 3.0 / duration)
    fwd_len = _safe_float(row, colmap["fwd_len"], 0.0)
    bwd_len = _safe_float(row, colmap["bwd_len"], 0.0)
    fwd_pkts = _safe_float(row, colmap["fwd_pkts"], 0.0)
    bwd_pkts = _safe_float(row, colmap["bwd_pkts"], 0.0)
    bytes_total = max(0.0, fwd_len + bwd_len)
    packets_total = max(0.0, fwd_pkts + bwd_pkts)
    flow_iat_mean = _safe_float(row, colmap["flow_iat_mean"], 0.0)
    flow_iat_std = _safe_float(row, colmap["flow_iat_std"], 0.0)
    pkt_len_mean = _safe_float(row, colmap["pkt_len_mean"], 0.0)
    pkt_len_std = _safe_float(row, colmap["pkt_len_std"], 0.0)
    syn_count = _safe_float(row, colmap["syn_count"], 0.0)
    raw_label = row[colmap["label"]] if colmap["label"] and pd.notna(row[colmap["label"]]) else fallback_label
    label = normalize_label(raw_label, fallback=fallback_label)

    estimated_packets = max(1.0, packet_rate * duration)
    if packets_total <= 0:
        packets_total = estimated_packets
    avg_packet_size = pkt_len_mean if pkt_len_mean > 0 else (bytes_total / max(1.0, packets_total))
    burstiness = max(1.0, packet_rate)
    fwd_to_bwd_ratio = (fwd_pkts + 1.0) / (bwd_pkts + 1.0)
    iat_cv = flow_iat_std / max(1e-6, flow_iat_mean) if flow_iat_mean > 0 else 0.0

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,
        "duration": duration,
        "packet_rate": packet_rate,
        "bytes_total": bytes_total,
        "syn_count": syn_count,
        "label": label,
        "estimated_packets": estimated_packets,
        "packets_total": packets_total,
        "avg_packet_size": avg_packet_size,
        "fwd_packets": fwd_pkts,
        "bwd_packets": bwd_pkts,
        "fwd_to_bwd_ratio": fwd_to_bwd_ratio,
        "flow_iat_mean": flow_iat_mean,
        "flow_iat_std": flow_iat_std,
        "flow_iat_cv": iat_cv,
        "packet_len_std": pkt_len_std,
        "burstiness": burstiness,
    }


def build_semantic_event_sequence(flow: Dict) -> List[str]:
    events: List[str] = []

    proto = flow["protocol"]
    dst_port = flow["dst_port"]
    syn_count = flow["syn_count"]
    duration = flow["duration"]
    packet_rate = flow["packet_rate"]
    bytes_total = flow["bytes_total"]
    burstiness = flow["burstiness"]
    avg_pkt = flow["avg_packet_size"]

    # Stage 1: protocol/handshake intent (keep coarse and stable)
    if proto == 17:
        events.append("UDP_FLOW_START")
    elif syn_count > 0:
        events.append("TCP_SYN_START")
    else:
        events.append("TCP_FLOW_START")

    # Stage 2: service context
    if dst_port == 53:
        events.append("DNS_SERVICE")
    elif dst_port in (80, 443):
        events.append("WEB_SERVICE")
    elif dst_port in (21, 22):
        events.append("AUTH_SERVICE")
    elif dst_port in (25, 110, 143, 587):
        events.append("MAIL_SERVICE")
    elif dst_port > 1024:
        events.append("HIGH_PORT_SERVICE")
    else:
        events.append("SYSTEM_PORT_SERVICE")

    # Stage 3: transfer profile (broader bins)
    if bytes_total > 2_000_000:
        events.append("XL_TRANSFER")
    elif bytes_total > 150_000:
        events.append("LARGE_TRANSFER")
    elif bytes_total < 1_500:
        events.append("TINY_TRANSFER")
    else:
        events.append("NORMAL_TRANSFER")

    # Stage 4: pacing profile (reduced sensitivity)
    if packet_rate > 1500:
        events.append("EXTREME_RATE")
    elif packet_rate > 400:
        events.append("HIGH_RATE")
    elif packet_rate < 8:
        events.append("LOW_RATE")
    else:
        events.append("NORMAL_RATE")

    # Stage 5: burst profile
    if burstiness > 2500:
        events.append("MICROBURST")
    elif burstiness > 700:
        events.append("BURSTY")
    else:
        events.append("STEADY_PACE")

    # Stage 6: behavior cue (explicit scan/flood semantics)
    syn_density = syn_count / max(1.0, flow["estimated_packets"])
    if syn_count >= 8 and syn_density > 0.3 and packet_rate > 60:
        events.append("SYN_SWEEP_HINT")
    elif packet_rate > 1200 and bytes_total < 20_000:
        events.append("PACKET_FLOOD_HINT")
    elif dst_port in (3389, 445) and packet_rate > 60:
        events.append("LATERAL_MOVE_HINT")
    elif duration > 60:
        events.append("LONG_FLOW")
    elif duration < 0.8:
        events.append("SHORT_FLOW")
    elif avg_pkt > 1250:
        events.append("HEAVY_PAYLOAD")
    else:
        events.append("STANDARD_FLOW")

    # Stage 7: application-level interaction cue (helps web/rare-family separability)
    packets_total = flow["packets_total"]
    fwd_to_bwd = flow["fwd_to_bwd_ratio"]
    iat_cv = flow["flow_iat_cv"]
    bwd_pkts = flow["bwd_packets"]

    if dst_port in (80, 443):
        if bytes_total < 2_000 and duration < 1.5 and packets_total <= 12:
            events.append("WEB_MICRO_REQUEST")
        elif fwd_to_bwd > 8.0 and bwd_pkts <= 1:
            events.append("WEB_ONE_WAY_BURST")
        elif iat_cv < 0.25 and packet_rate > 120:
            events.append("WEB_AUTOMATION_RHYTHM")
        elif avg_pkt < 220 and packets_total > 30:
            events.append("WEB_TINY_PACKET_SPRAY")
        else:
            events.append("WEB_NORMAL_EXCHANGE")
    elif fwd_to_bwd > 6.0 and bwd_pkts <= 1:
        events.append("ASYMMETRIC_PUSH")
    elif iat_cv > 2.5 and duration > 5:
        events.append("IRREGULAR_TIMING")
    else:
        events.append("BALANCED_EXCHANGE")

    # Stage 8: close
    events.append("SESSION_END")
    return events


def build_session_rows(session_id: str, flow: Dict) -> List[Dict]:
    events = build_semantic_event_sequence(flow)
    n = len(events)
    if n < 2:
        events = events + ["SESSION_END"]
        n = len(events)

    step = flow["duration"] / max(1, n - 1)
    packet_len = flow["bytes_total"] / n if flow["bytes_total"] > 0 else 0.0

    base = {
        "label": flow["label"],
        "session_duration": flow["duration"],
        "session_bytes_total": flow["bytes_total"],
        "session_packet_rate": flow["packet_rate"],
        "session_packets_total": int(max(1.0, round(flow["packets_total"]))),
        "session_unique_dst_ips": 1,
        "session_unique_dst_ports": 1,
        "session_max_packets_per_sec": flow["burstiness"],
        "length": packet_len,
        "dst_port": flow["dst_port"],
        "protocol": flow["protocol"],
        "syn_flag_count": flow["syn_count"],
    }

    rows: List[Dict] = []
    for i, event in enumerate(events):
        rows.append({
            "session_id": session_id,
            "timestamp": float(i * step),
            "event": event,
            **base,
        })
    return rows

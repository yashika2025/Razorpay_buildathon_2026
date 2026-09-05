"""
security_analysis.py
----------------------
The second half of the cybersecurity/integrity layer: entity correlation.

anomaly_detection.py looks at each transaction's own adjustment pattern.
This module instead looks ACROSS transactions, at the audit_log table, to
answer a different question: "is the same operator, device, or IP address
showing up in an unusual pattern across many records?"

This is the classic security-analyst move of pivoting on an indicator
(an operator ID, a device ID, an IP address) rather than looking at any one
event in isolation -- a single manual adjustment looks fine on its own, but
one operator touching 45 unrelated transactions from one device in a single
burst is a pattern only visible at the entity level.

We use a simple graph (NetworkX) with three node types -- operator, device,
IP -- connected via the audit log records they appear together in. We do
NOT build a complex fraud-ring graph model; we use the graph only to compute
straightforward relationship counts (a operator-device-IP correlation
table), which keeps this explainable and appropriately scoped for a
hackathon timeline.

Run directly: python src/security_analysis.py
"""

import os
import pandas as pd
import networkx as nx

import config as cfg

PROJECT_ROOT = cfg.PROJECT_ROOT
RAW_DIR = cfg.RAW_DIR
PROCESSED_DIR = cfg.PROCESSED_DIR

# An operator touching more transactions than this within the dataset's
# time window is flagged as a high-volume outlier for manual review.
# Uses a MAD-based robust z-score (see compute_operator_volume_outliers) --
# 3.5 is a common convention for robust z-score outlier thresholds, chosen
# because MAD-based scores run higher than standard mean/std z-scores.
HIGH_VOLUME_ZSCORE = cfg.HIGH_VOLUME_ZSCORE

# A device or IP shared across this many distinct operators is unusual --
# in normal operations, one device/IP maps to roughly one operator.
SHARED_DEVICE_OPERATOR_THRESHOLD = cfg.SHARED_DEVICE_OPERATOR_THRESHOLD


def load_audit_logs():
    path = os.path.join(RAW_DIR, "audit_logs.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def build_entity_graph(audit_logs):
    """
    Builds an undirected graph with operator/device/IP nodes. An edge is
    added between an operator and the device/IP they used, weighted by how
    many audit log events that pair appears together in.
    """
    G = nx.Graph()
    if audit_logs.empty:
        return G

    for _, row in audit_logs.iterrows():
        op = f"operator::{row['operator_id']}"
        dev = f"device::{row['device_id']}"
        ip = f"ip::{row['ip_address']}"

        for node in (op, dev, ip):
            if not G.has_node(node):
                G.add_node(node)

        for a, b in [(op, dev), (op, ip), (dev, ip)]:
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    return G


def compute_operator_volume_outliers(audit_logs):
    """Flags operators whose total number of DISTINCT entities touched is
    a statistical outlier relative to their peers."""
    if audit_logs.empty:
        return pd.DataFrame(columns=["operator_id", "entities_touched", "zscore", "is_outlier"])

    counts = audit_logs.groupby("operator_id")["entity_id"].nunique().reset_index()
    counts.columns = ["operator_id", "entities_touched"]

    median = counts["entities_touched"].median()
    mad = (counts["entities_touched"] - median).abs().median()
    mad = mad if mad > 0 else 1.0
    counts["zscore"] = (0.6745 * (counts["entities_touched"] - median) / mad).round(2)
    counts["is_outlier"] = counts["zscore"] >= HIGH_VOLUME_ZSCORE
    return counts.sort_values("zscore", ascending=False)


def compute_shared_device_ip(audit_logs):
    """Flags devices/IPs used by more than one operator -- in a normal
    workflow each operator has their own device/IP; a shared device/IP
    across operators can indicate a compromised endpoint or shared/misused
    credentials."""
    if audit_logs.empty:
        return pd.DataFrame(), pd.DataFrame()

    device_ops = audit_logs.groupby("device_id")["operator_id"].nunique().reset_index()
    device_ops.columns = ["device_id", "distinct_operators"]
    device_ops["is_shared"] = device_ops["distinct_operators"] >= SHARED_DEVICE_OPERATOR_THRESHOLD

    ip_ops = audit_logs.groupby("ip_address")["operator_id"].nunique().reset_index()
    ip_ops.columns = ["ip_address", "distinct_operators"]
    ip_ops["is_shared"] = ip_ops["distinct_operators"] >= SHARED_DEVICE_OPERATOR_THRESHOLD

    return device_ops.sort_values("distinct_operators", ascending=False), \
        ip_ops.sort_values("distinct_operators", ascending=False)


def build_transaction_level_security_flags(audit_logs, operator_outliers, device_flags, ip_flags):
    """
    Projects the entity-level findings back onto individual transactions,
    so the risk scorer (risk_scoring.py) can add a "this transaction was
    touched by a high-volume/outlier operator or a shared device/IP" signal
    per transaction_id, not just per entity.
    """
    if audit_logs.empty:
        return pd.DataFrame(columns=["transaction_id", "security_flags"])

    outlier_ops = set(operator_outliers.loc[operator_outliers["is_outlier"], "operator_id"])
    shared_devices = set(device_flags.loc[device_flags["is_shared"], "device_id"]) if not device_flags.empty else set()
    shared_ips = set(ip_flags.loc[ip_flags["is_shared"], "ip_address"]) if not ip_flags.empty else set()

    results = {}
    for _, row in audit_logs.iterrows():
        txn_id = row["entity_id"]
        flags = results.setdefault(txn_id, set())
        if row["operator_id"] in outlier_ops:
            flags.add(f"touched by high-volume operator {row['operator_id']}")
        if row["device_id"] in shared_devices:
            flags.add(f"device {row['device_id']} shared across multiple operators")
        if row["ip_address"] in shared_ips:
            flags.add(f"IP {row['ip_address']} shared across multiple operators")

    rows = [{"transaction_id": txn, "security_flags": " | ".join(sorted(flags))}
            for txn, flags in results.items() if flags]
    return pd.DataFrame(rows)


def run_security_analysis():
    audit_logs = load_audit_logs()
    graph = build_entity_graph(audit_logs)
    operator_outliers = compute_operator_volume_outliers(audit_logs)
    device_flags, ip_flags = compute_shared_device_ip(audit_logs)
    txn_flags = build_transaction_level_security_flags(audit_logs, operator_outliers, device_flags, ip_flags)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    operator_outliers.to_csv(os.path.join(PROCESSED_DIR, "operator_outliers.csv"), index=False)
    txn_flags.to_csv(os.path.join(PROCESSED_DIR, "security_flags.csv"), index=False)

    print(f"Entity graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"\nOperator activity outliers (z-score >= {HIGH_VOLUME_ZSCORE}):")
    print(operator_outliers[operator_outliers["is_outlier"]])
    print(f"\nTransactions with entity-level security flags: {len(txn_flags)}")
    print(f"Saved to data/processed/operator_outliers.csv and data/processed/security_flags.csv")

    return operator_outliers, device_flags, ip_flags, txn_flags


if __name__ == "__main__":
    run_security_analysis()

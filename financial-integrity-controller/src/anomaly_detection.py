"""
anomaly_detection.py
---------------------
The first half of the cybersecurity/integrity layer.

IMPORTANT DESIGN DECISION:
This module does NOT only look at financially "UNRESOLVED" transactions.
As the reconciliation results show, a manipulated or abnormally-modified
transaction can still be financially self-consistent (the adjustment record
correctly explains the settlement gap) while still being behaviorally
suspicious (odd hours, unusual frequency, unusual operator). So anomaly
detection runs on BEHAVIORAL features derived from adjustments and audit
logs, for every transaction that has any adjustment/audit activity at all,
regardless of its reconciliation status.

We use two complementary approaches, on purpose:

1. STATISTICAL / RULE-BASED signals -- transparent, explainable, catches
   the "obvious" cases (odd hours, high frequency, large single adjustment).
2. Isolation Forest -- an unsupervised ML model that can catch combinations
   of features that no single rule would flag, at the cost of being less
   directly explainable (we still extract feature contributions to make it
   as explainable as reasonably possible).

WHY NOT SUPERVISED CLASSIFICATION:
We have no reliable ground-truth "this was fraud" label in a real deployment
(our synthetic scenario_labels.csv is a research/evaluation convenience, not
something a production system would have). Isolation Forest is designed for
exactly this: finding rare, different-looking points without needing labels.

Run directly: python src/anomaly_detection.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config as cfg

PROJECT_ROOT = cfg.PROJECT_ROOT
PROCESSED_DIR = cfg.PROCESSED_DIR
RAW_DIR = cfg.RAW_DIR

ODD_HOURS = set(range(*cfg.ODD_HOURS_RANGE))  # midnight - 6am counts as "unusual hours"

# Rule-based thresholds (documented so they can be defended in a panel).
HIGH_FREQUENCY_ADJUSTMENT_COUNT = cfg.HIGH_FREQUENCY_ADJUSTMENT_COUNT
LARGE_SINGLE_ADJUSTMENT_FRACTION = cfg.LARGE_SINGLE_ADJUSTMENT_FRACTION
SHORT_WINDOW_MINUTES = cfg.SHORT_WINDOW_MINUTES


def load_inputs():
    exceptions_path = os.path.join(PROCESSED_DIR, "exceptions.csv")
    if not os.path.exists(exceptions_path):
        raise FileNotFoundError("exceptions.csv not found -- run exceptions.py first.")

    recon = pd.read_csv(exceptions_path)
    adjustments = pd.read_csv(os.path.join(RAW_DIR, "adjustments.csv"))
    audit_logs = pd.read_csv(os.path.join(RAW_DIR, "audit_logs.csv"))
    return recon, adjustments, audit_logs


def build_behavioral_features(recon, adjustments, audit_logs):
    """
    Builds one feature row per transaction that had ANY adjustment or audit
    log activity. Transactions with no such activity are not candidates for
    behavioral anomaly detection (there's no behavior to analyze) and are
    excluded here -- they may still be financial exceptions, just not
    security-relevant ones.
    """
    if adjustments.empty:
        return pd.DataFrame()

    adjustments = adjustments.copy()
    adjustments["timestamp"] = pd.to_datetime(adjustments["timestamp"])
    adjustments["hour"] = adjustments["timestamp"].dt.hour

    features = []
    for txn_id, group in adjustments.groupby("transaction_id"):
        group = group.sort_values("timestamp")
        payment_amount = recon.loc[recon["transaction_id"] == txn_id, "amount"]
        payment_amount = float(payment_amount.iloc[0]) if len(payment_amount) else np.nan

        n_adjustments = len(group)
        total_adjustment = group["adjustment_amount"].sum()
        max_single_adjustment = group["adjustment_amount"].max()
        n_odd_hour = group["hour"].isin(ODD_HOURS).sum()
        n_distinct_operators = group["operator_id"].nunique()

        # time span between first and last adjustment, in minutes
        span_minutes = (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 60.0

        large_single_fraction = (
            max_single_adjustment / payment_amount if payment_amount and payment_amount > 0 else 0
        )

        features.append({
            "transaction_id": txn_id,
            "operator_id": group["operator_id"].mode().iloc[0],  # most common operator for this txn
            "n_adjustments": n_adjustments,
            "total_adjustment": round(total_adjustment, 2),
            "max_single_adjustment": round(max_single_adjustment, 2),
            "large_single_fraction": round(large_single_fraction, 3),
            "n_odd_hour_adjustments": int(n_odd_hour),
            "n_distinct_operators": n_distinct_operators,
            "adjustment_span_minutes": round(span_minutes, 2),
            "is_rapid_burst": bool(n_adjustments > 1 and span_minutes <= SHORT_WINDOW_MINUTES),
        })

    feat_df = pd.DataFrame(features)

    # Add operator-level behavioral baseline: how many DISTINCT transactions
    # has this operator touched overall, across adjustments AND audit logs?
    # A normal operator touches a handful; a compromised/malicious one touches
    # far more than their peers.
    op_adj_counts = adjustments.groupby("operator_id")["transaction_id"].nunique()
    op_audit_counts = (
        audit_logs.groupby("operator_id")["entity_id"].nunique()
        if not audit_logs.empty else pd.Series(dtype=int)
    )
    combined_counts = op_adj_counts.add(op_audit_counts, fill_value=0)

    # We use a MAD-based (median absolute deviation) robust z-score here
    # instead of a standard mean/std z-score. Reason: standard deviation
    # itself gets inflated by extreme outliers (e.g. one compromised
    # operator touching 125 transactions), which can mask a SECOND abnormal
    # operator whose count is merely "very high" rather than "extreme".
    # The median and MAD are far less sensitive to a small number of
    # extreme points, so both abnormal operators surface clearly.
    median = combined_counts.median()
    mad = (combined_counts - median).abs().median()
    mad = mad if mad > 0 else 1.0
    # 0.6745 scales MAD to be comparable to a standard-deviation z-score
    # under a normal distribution -- a common convention for robust z-scores.
    robust_zscore = 0.6745 * (combined_counts - median) / mad

    feat_df["operator_total_txns_touched"] = feat_df["operator_id"].map(combined_counts).fillna(0)
    feat_df["operator_activity_zscore"] = feat_df["operator_id"].map(robust_zscore).fillna(0).round(2)

    return feat_df


def apply_rule_based_flags(feat_df):
    """Transparent, human-readable rule checks. Each produces a boolean flag
    and a short reason string, which is far easier to defend in a panel than
    a raw ML score alone."""
    reasons_col = []
    flags_col = []

    for _, row in feat_df.iterrows():
        reasons = []
        if row["n_adjustments"] >= HIGH_FREQUENCY_ADJUSTMENT_COUNT:
            reasons.append(f"{int(row['n_adjustments'])} manual adjustments on a single transaction")
        if row["is_rapid_burst"]:
            reasons.append(f"multiple adjustments within {SHORT_WINDOW_MINUTES} minutes")
        if row["n_odd_hour_adjustments"] > 0:
            reasons.append(f"{int(row['n_odd_hour_adjustments'])} adjustment(s) made between midnight-6am")
        if row["large_single_fraction"] >= LARGE_SINGLE_ADJUSTMENT_FRACTION:
            reasons.append(f"single adjustment equal to {row['large_single_fraction']*100:.0f}% of payment amount")
        if row["operator_activity_zscore"] >= cfg.OPERATOR_ZSCORE_FLAG_THRESHOLD:
            reasons.append(
                f"operator {row['operator_id']} has touched {int(row['operator_total_txns_touched'])} "
                f"transactions, far above the typical operator baseline"
            )
        reasons_col.append(reasons)
        flags_col.append(len(reasons) > 0)

    feat_df = feat_df.copy()
    feat_df["rule_flags"] = reasons_col
    feat_df["rule_flag_count"] = feat_df["rule_flags"].apply(len)
    feat_df["rule_based_anomaly"] = flags_col
    return feat_df


ML_FEATURE_COLUMNS = [
    "n_adjustments", "total_adjustment", "max_single_adjustment",
    "large_single_fraction", "n_odd_hour_adjustments", "n_distinct_operators",
    "adjustment_span_minutes", "operator_total_txns_touched", "operator_activity_zscore",
]


def apply_isolation_forest(feat_df, contamination=cfg.ISOLATION_FOREST_CONTAMINATION,
                            random_state=cfg.ISOLATION_FOREST_RANDOM_STATE):
    """
    Fits an unsupervised Isolation Forest over the behavioral features.
    contamination=0.1 means we expect roughly 10% of transactions-with-
    activity to be outliers -- a deliberately loose upper bound so the model
    surfaces candidates for the risk scorer to prioritize, rather than
    trying to make a final yes/no call itself.
    """
    if len(feat_df) < 10:
        # too few samples for a meaningful model; skip ML scoring
        feat_df = feat_df.copy()
        feat_df["ml_anomaly_score"] = 0.0
        feat_df["ml_is_outlier"] = False
        return feat_df

    X = feat_df[ML_FEATURE_COLUMNS].fillna(0).values
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)

    # decision_function: higher = more normal, lower/negative = more anomalous.
    # We flip and rescale to a 0-1 "anomaly score" so it's intuitive: higher = more anomalous.
    raw_scores = model.decision_function(X)
    normalized = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)

    feat_df = feat_df.copy()
    feat_df["ml_anomaly_score"] = normalized.round(3)
    feat_df["ml_is_outlier"] = model.predict(X) == -1  # -1 = outlier in sklearn's convention
    return feat_df


def run_anomaly_detection():
    recon, adjustments, audit_logs = load_inputs()
    feat_df = build_behavioral_features(recon, adjustments, audit_logs)

    if feat_df.empty:
        print("No adjustment activity found -- nothing to analyze behaviorally.")
        return feat_df

    feat_df = apply_rule_based_flags(feat_df)
    feat_df = apply_isolation_forest(feat_df)

    out_path = os.path.join(PROCESSED_DIR, "anomaly_features.csv")
    # rule_flags is a list -- store as a readable string for CSV
    feat_df_out = feat_df.copy()
    feat_df_out["rule_flags"] = feat_df_out["rule_flags"].apply(lambda r: " | ".join(r))
    feat_df_out.to_csv(out_path, index=False)

    print(f"Analyzed {len(feat_df)} transactions with adjustment/audit activity.")
    print(f"  Rule-based anomalies: {feat_df['rule_based_anomaly'].sum()}")
    print(f"  ML (Isolation Forest) outliers: {feat_df['ml_is_outlier'].sum()}")
    print(f"\nSaved to {out_path}")
    return feat_df


if __name__ == "__main__":
    run_anomaly_detection()

"""
risk_scoring.py
-----------------
Combines the outputs of exceptions.py, anomaly_detection.py, and
security_analysis.py into one transparent, additive risk score per
transaction, plus a plain-English list of reasons.

WHY ADDITIVE AND NOT A BLACK BOX:
A finance/security reviewer needs to be able to look at a score of 87 and
know exactly which four things added up to it. So the score is built as a
sum of clearly-named point contributions, each with a documented reason
string. Nothing here is "off the top of my head" -- every point value below
is deliberately chosen and explained in the comments so it can be discussed
in a technical panel.

SCORE COMPONENTS (each capped so no single factor can dominate unfairly):
  1. Financial discrepancy severity   (0-30 points)
  2. Behavioral anomaly (rule-based)  (0-30 points)
  3. ML anomaly score (Isolation Forest) (0-20 points)
  4. Entity correlation / security flags (0-20 points)

Note that a transaction with ZERO financial discrepancy (fully reconciled
on paper) can still reach HIGH risk purely on behavioral + ML + entity
signals. This is intentional: a compromised or malicious operator who
correctly records their own adjustment leaves the books balanced, but the
*pattern* of how and when they did it is still the strongest signal we
have that something needs human review.

Total range: 0-100.

Risk levels:
  0-29   -> LOW
  30-59  -> MEDIUM
  60-100 -> HIGH

Run directly: python src/risk_scoring.py
"""

import os
import pandas as pd

import config as cfg

PROJECT_ROOT = cfg.PROJECT_ROOT
PROCESSED_DIR = cfg.PROCESSED_DIR

LOW_MEDIUM_CUTOFF = cfg.RISK_LOW_MEDIUM_CUTOFF
MEDIUM_HIGH_CUTOFF = cfg.RISK_MEDIUM_HIGH_CUTOFF


def load_all():
    exceptions = pd.read_csv(os.path.join(PROCESSED_DIR, "exceptions.csv"))

    anomaly_path = os.path.join(PROCESSED_DIR, "anomaly_features.csv")
    anomaly = pd.read_csv(anomaly_path) if os.path.exists(anomaly_path) else pd.DataFrame()

    security_path = os.path.join(PROCESSED_DIR, "security_flags.csv")
    security = pd.read_csv(security_path) if os.path.exists(security_path) else pd.DataFrame()

    return exceptions, anomaly, security


def score_financial_discrepancy(row, max_points=cfg.RISK_WEIGHT_FINANCIAL):
    """
    Only transactions that are actual exceptions (not MATCHED or
    RECONCILED_WITH_ADJUSTMENT) earn points here. The size of the
    unexplained amount, relative to the original payment amount, determines
    how many points -- a ₹50 gap on a ₹20,000 payment is very different from
    a ₹15,000 gap on the same payment.
    """
    status = row["recon_status"]
    if status in ("MATCHED", "RECONCILED_WITH_ADJUSTMENT"):
        return 0, []

    reasons = []
    points = 0

    if status == "MISSING_SETTLEMENT":
        points = max_points * 0.6
        reasons.append("Settlement record is missing entirely")
    elif status == "DUPLICATE_SETTLEMENT":
        points = max_points * 0.5
        reasons.append(f"{int(row['settlement_count'])} duplicate settlement records found")
    else:  # UNRESOLVED or PARTIALLY_EXPLAINED
        diff = row["difference"]
        amount = row["amount"] if row["amount"] else 1
        severity_ratio = min(abs(diff) / amount, 1) if pd.notna(diff) else 0.5
        points = max_points * severity_ratio
        if pd.notna(diff):
            reasons.append(f"Unexplained discrepancy of \u20b9{abs(diff):,.2f} ({severity_ratio*100:.1f}% of payment amount)")

    return round(min(points, max_points), 1), reasons


def score_behavioral_anomaly(anomaly_row, max_points=cfg.RISK_WEIGHT_BEHAVIORAL):
    if anomaly_row is None:
        return 0, []
    points = 0
    reasons = []
    flag_count = anomaly_row.get("rule_flag_count", 0)
    if flag_count > 0:
        # each rule flag contributes points, capped at max_points
        per_flag = max_points / 5.0  # designed so 5 simultaneous flags = max score
        points = min(flag_count * per_flag, max_points)
        flags_str = anomaly_row.get("rule_flags", "")
        if isinstance(flags_str, str) and flags_str:
            reasons.extend(flags_str.split(" | "))
    return round(points, 1), reasons


def score_ml_anomaly(anomaly_row, max_points=cfg.RISK_WEIGHT_ML):
    if anomaly_row is None:
        return 0, []
    ml_score = anomaly_row.get("ml_anomaly_score", 0) or 0
    is_outlier = anomaly_row.get("ml_is_outlier", False)
    points = round(ml_score * max_points, 1)
    reasons = []
    if is_outlier:
        reasons.append(
            f"Flagged as a statistical outlier by the anomaly detection model "
            f"(anomaly score {ml_score:.2f})"
        )
    return points, reasons


def score_entity_correlation(security_flags_str, max_points=cfg.RISK_WEIGHT_ENTITY):
    if not isinstance(security_flags_str, str) or not security_flags_str:
        return 0, []
    flags = security_flags_str.split(" | ")
    per_flag = max_points / 2.0  # 2 simultaneous entity flags = max score
    points = min(len(flags) * per_flag, max_points)
    return round(points, 1), flags


def classify_risk_level(score):
    if score >= MEDIUM_HIGH_CUTOFF:
        return "HIGH"
    elif score >= LOW_MEDIUM_CUTOFF:
        return "MEDIUM"
    return "LOW"


def recommend_action(risk_level, recon_status):
    if risk_level == "HIGH":
        return "Manual investigation required"
    if risk_level == "MEDIUM":
        return "Review recommended"
    if recon_status in ("MATCHED", "RECONCILED_WITH_ADJUSTMENT"):
        return "No action needed"
    return "Monitor"


def compute_risk_scores():
    exceptions, anomaly, security = load_all()

    anomaly_by_txn = anomaly.set_index("transaction_id").to_dict("index") if not anomaly.empty else {}
    security_by_txn = (
        security.set_index("transaction_id")["security_flags"].to_dict() if not security.empty else {}
    )

    results = []
    for _, row in exceptions.iterrows():
        txn_id = row["transaction_id"]

        fin_points, fin_reasons = score_financial_discrepancy(row)
        anomaly_row = anomaly_by_txn.get(txn_id)
        behav_points, behav_reasons = score_behavioral_anomaly(anomaly_row)
        ml_points, ml_reasons = score_ml_anomaly(anomaly_row)
        entity_points, entity_reasons = score_entity_correlation(security_by_txn.get(txn_id))

        total = round(fin_points + behav_points + ml_points + entity_points, 1)
        total = min(total, 100.0)
        risk_level = classify_risk_level(total)
        all_reasons = fin_reasons + behav_reasons + ml_reasons + entity_reasons

        results.append({
            "transaction_id": txn_id,
            "recon_status": row["recon_status"],
            "financial_points": fin_points,
            "behavioral_points": behav_points,
            "ml_points": ml_points,
            "entity_points": entity_points,
            "risk_score": total,
            "risk_level": risk_level,
            "recommended_action": recommend_action(risk_level, row["recon_status"]),
            "reasons": " | ".join(all_reasons) if all_reasons else "No risk indicators found",
        })

    risk_df = pd.DataFrame(results)
    merged = exceptions.merge(risk_df.drop(columns=["recon_status"]), on="transaction_id", how="left")

    out_path = os.path.join(PROCESSED_DIR, "risk_scores.csv")
    merged.to_csv(out_path, index=False)

    print("Risk scoring complete.")
    print(merged["risk_level"].value_counts())
    print(f"\nSaved to {out_path}")

    high_risk = merged[merged["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False)
    if len(high_risk) > 0:
        print(f"\nTop HIGH-risk transactions:")
        for _, r in high_risk.head(5).iterrows():
            print(f"  {r['transaction_id']}  score={r['risk_score']}  -- {r['reasons']}")

    return merged


if __name__ == "__main__":
    compute_risk_scores()

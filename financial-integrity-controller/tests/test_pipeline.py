"""
test_pipeline.py
------------------
Tests for the reconciliation, exception classification, and anomaly
detection logic. These use small, hand-built DataFrames (not the full
synthetic dataset) so each test is self-contained and its expected outcome
is obvious just by reading the test.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_pipeline.py   (falls back to plain asserts)
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from reconciliation import build_reconciliation_table
from exceptions import classify_row, classify_all


def make_tables(payments=None, settlements=None, refunds=None, fees=None,
                 taxes=None, adjustments=None, audit_logs=None):
    """Helper to build a minimal `tables` dict like data_validator.load_raw_tables
    would, defaulting any un-provided table to an empty DataFrame."""
    return {
        "payments": pd.DataFrame(payments or []),
        "settlements": pd.DataFrame(settlements or []),
        "refunds": pd.DataFrame(refunds or []),
        "fees": pd.DataFrame(fees or []),
        "taxes": pd.DataFrame(taxes or []),
        "adjustments": pd.DataFrame(adjustments or []),
        "audit_logs": pd.DataFrame(audit_logs or []),
    }


def classify_txn(tables):
    """Runs reconciliation + classification and returns the single-row
    result as a dict for easy assertions."""
    recon = build_reconciliation_table(tables)
    classified = classify_all(recon)
    return classified.iloc[0].to_dict()


# ----------------------------------------------------------------------
# 1. Exact match
# ----------------------------------------------------------------------
def test_exact_match():
    tables = make_tables(
        payments=[{"transaction_id": "T1", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
        settlements=[{"settlement_id": "S1", "transaction_id": "T1", "settlement_amount": 10000,
                      "settlement_timestamp": "2026-01-02T10:00:00", "settlement_status": "SETTLED"}],
    )
    result = classify_txn(tables)
    assert result["recon_status"] == "MATCHED", result
    print("PASS: test_exact_match")


# ----------------------------------------------------------------------
# 2. Fee-adjusted match
# ----------------------------------------------------------------------
def test_fee_adjusted_match():
    tables = make_tables(
        payments=[{"transaction_id": "T2", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
        settlements=[{"settlement_id": "S2", "transaction_id": "T2", "settlement_amount": 9800,
                      "settlement_timestamp": "2026-01-02T10:00:00", "settlement_status": "SETTLED"}],
        fees=[{"fee_id": "F1", "transaction_id": "T2", "fee_amount": 200}],
    )
    result = classify_txn(tables)
    assert result["recon_status"] == "RECONCILED_WITH_ADJUSTMENT", result
    print("PASS: test_fee_adjusted_match")


# ----------------------------------------------------------------------
# 3. Tax-adjusted match
# ----------------------------------------------------------------------
def test_tax_adjusted_match():
    tables = make_tables(
        payments=[{"transaction_id": "T3", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
        settlements=[{"settlement_id": "S3", "transaction_id": "T3", "settlement_amount": 9964,
                      "settlement_timestamp": "2026-01-02T10:00:00", "settlement_status": "SETTLED"}],
        taxes=[{"tax_id": "TX1", "transaction_id": "T3", "tax_amount": 36}],
    )
    result = classify_txn(tables)
    assert result["recon_status"] == "RECONCILED_WITH_ADJUSTMENT", result
    print("PASS: test_tax_adjusted_match")


# ----------------------------------------------------------------------
# 4. Missing settlement
# ----------------------------------------------------------------------
def test_missing_settlement():
    tables = make_tables(
        payments=[{"transaction_id": "T4", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
    )
    result = classify_txn(tables)
    assert result["recon_status"] == "MISSING_SETTLEMENT", result
    print("PASS: test_missing_settlement")


# ----------------------------------------------------------------------
# 5. Duplicate settlement
# ----------------------------------------------------------------------
def test_duplicate_settlement():
    tables = make_tables(
        payments=[{"transaction_id": "T5", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
        settlements=[
            {"settlement_id": "S5A", "transaction_id": "T5", "settlement_amount": 10000,
             "settlement_timestamp": "2026-01-02T10:00:00", "settlement_status": "SETTLED"},
            {"settlement_id": "S5B", "transaction_id": "T5", "settlement_amount": 10000,
             "settlement_timestamp": "2026-01-02T12:00:00", "settlement_status": "SETTLED"},
        ],
    )
    result = classify_txn(tables)
    assert result["recon_status"] == "DUPLICATE_SETTLEMENT", result
    print("PASS: test_duplicate_settlement")


# ----------------------------------------------------------------------
# 6. Incorrect amount (operational error, unresolved)
# ----------------------------------------------------------------------
def test_incorrect_amount():
    tables = make_tables(
        payments=[{"transaction_id": "T6", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
        settlements=[{"settlement_id": "S6", "transaction_id": "T6", "settlement_amount": 8500,
                      "settlement_timestamp": "2026-01-02T10:00:00", "settlement_status": "SETTLED"}],
        # no fee/tax/refund/adjustment recorded -- ₹1500 is entirely unexplained
    )
    result = classify_txn(tables)
    assert result["recon_status"] == "UNRESOLVED", result
    assert abs(result["difference"] - (-1500)) < 0.01, result
    print("PASS: test_incorrect_amount")


# ----------------------------------------------------------------------
# 7. Duplicate refund (double-counted, becomes unexplained)
# ----------------------------------------------------------------------
def test_duplicate_refund():
    tables = make_tables(
        payments=[{"transaction_id": "T7", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
        settlements=[{"settlement_id": "S7", "transaction_id": "T7", "settlement_amount": 9000,
                      "settlement_timestamp": "2026-01-02T10:00:00", "settlement_status": "SETTLED"}],
        refunds=[
            {"refund_id": "R7A", "transaction_id": "T7", "refund_amount": 1000, "refund_timestamp": "2026-01-02T11:00:00"},
            {"refund_id": "R7B", "transaction_id": "T7", "refund_amount": 1000, "refund_timestamp": "2026-01-02T11:05:00"},
        ],
    )
    result = classify_txn(tables)
    # Actual settlement only reflects ONE refund (9000), but total_refund sums to 2000,
    # so expected_settlement = 10000-2000=8000, actual=9000 -> unresolved gap of +1000
    assert result["recon_status"] == "UNRESOLVED", result
    print("PASS: test_duplicate_refund")


# ----------------------------------------------------------------------
# 8. Unexplained discrepancy (partially explained scenario)
# ----------------------------------------------------------------------
def test_partially_explained():
    tables = make_tables(
        payments=[{"transaction_id": "T8", "merchant_id": "M1", "amount": 10000,
                   "currency": "INR", "timestamp": "2026-01-01T10:00:00", "status": "SUCCESS"}],
        settlements=[{"settlement_id": "S8", "transaction_id": "T8", "settlement_amount": 9200,
                      "settlement_timestamp": "2026-01-02T10:00:00", "settlement_status": "SETTLED"}],
        fees=[{"fee_id": "F8", "transaction_id": "T8", "fee_amount": 500}],
        # raw gap = 10000-9200=800; fee explains 500 of it (62.5%) -> partially explained
    )
    result = classify_txn(tables)
    assert result["recon_status"] == "PARTIALLY_EXPLAINED", result
    print("PASS: test_partially_explained")


# ----------------------------------------------------------------------
# 9 & 10. Suspicious modification / abnormal operator behavior
# (tested at the anomaly_detection.py level, since that's where behavioral
# signals are computed, not reconciliation)
# ----------------------------------------------------------------------
def test_suspicious_modification_flagged():
    from anomaly_detection import build_behavioral_features, apply_rule_based_flags

    recon = pd.DataFrame([
        {"transaction_id": "T9", "amount": 10000},
    ])
    adjustments = pd.DataFrame([
        {"transaction_id": "T9", "adjustment_amount": 6000, "operator_id": "OPX",
         "timestamp": "2026-01-01T02:30:00"},  # odd hour, large fraction of payment
    ])
    audit_logs = pd.DataFrame(columns=["operator_id", "entity_id"])

    feat = build_behavioral_features(recon, adjustments, audit_logs)
    feat = apply_rule_based_flags(feat)
    row = feat.iloc[0]
    assert row["rule_based_anomaly"] is True or row["rule_based_anomaly"] == True, row
    assert row["n_odd_hour_adjustments"] == 1
    assert row["large_single_fraction"] >= 0.3
    print("PASS: test_suspicious_modification_flagged")


def test_abnormal_operator_behavior_flagged():
    from anomaly_detection import build_behavioral_features, apply_rule_based_flags

    # one operator (OPY) touches many transactions; others touch few
    recon = pd.DataFrame([{"transaction_id": f"T{i}", "amount": 5000} for i in range(20)])
    adjustments_rows = []
    for i in range(15):  # OPY touches 15 of the 20 transactions
        adjustments_rows.append({"transaction_id": f"T{i}", "adjustment_amount": 100,
                                  "operator_id": "OPY", "timestamp": "2026-01-01T10:00:00"})
    for i in range(15, 18):  # OPZ touches 3
        adjustments_rows.append({"transaction_id": f"T{i}", "adjustment_amount": 100,
                                  "operator_id": "OPZ", "timestamp": "2026-01-01T10:00:00"})
    adjustments = pd.DataFrame(adjustments_rows)
    audit_logs = pd.DataFrame(columns=["operator_id", "entity_id"])

    feat = build_behavioral_features(recon, adjustments, audit_logs)
    feat = apply_rule_based_flags(feat)
    opy_rows = feat[feat["operator_id"] == "OPY"]
    assert (opy_rows["operator_activity_zscore"] > 0).all(), opy_rows
    print("PASS: test_abnormal_operator_behavior_flagged")


ALL_TESTS = [
    test_exact_match, test_fee_adjusted_match, test_tax_adjusted_match,
    test_missing_settlement, test_duplicate_settlement, test_incorrect_amount,
    test_duplicate_refund, test_partially_explained,
    test_suspicious_modification_flagged, test_abnormal_operator_behavior_flagged,
]

if __name__ == "__main__":
    failures = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test_fn.__name__} -- {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR: {test_fn.__name__} -- {e}")

    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tests passed.")
    sys.exit(1 if failures else 0)

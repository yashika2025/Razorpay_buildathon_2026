"""
reconciliation.py
------------------
The core Finance Controller engine.

For every payment, this computes what the settlement amount SHOULD be
(payment amount, minus any fees/taxes/refunds/adjustments on record), then
compares that expected amount against what was ACTUALLY settled.

This is deliberately deterministic and rule-based -- no ML here. Financial
reconciliation is a well-defined arithmetic problem; using a black-box model
for it would make the result unauditable, which is the opposite of what a
finance controller needs.

Output: one row per transaction with expected amount, actual amount,
difference, and a reconciliation_status (handled by exceptions.py, which
consumes this module's output).

Run directly: python src/reconciliation.py
"""

import pandas as pd

import config as cfg

# Amounts within this tolerance are treated as fully matched (protects
# against floating point rounding noise, not a business rule).
AMOUNT_TOLERANCE = cfg.AMOUNT_TOLERANCE


def aggregate_adjustments(tables):
    """
    Sums fees, taxes, refunds, and manual adjustments per transaction_id.
    A transaction can have zero, one, or many rows in each of these tables
    (e.g. repeated_manual_adjustments has many adjustment rows for one txn) --
    so we group and sum rather than assuming a 1:1 relationship.
    """
    def sum_by_txn(df, amount_col, out_col):
        if df.empty or amount_col not in df.columns:
            return pd.DataFrame(columns=["transaction_id", out_col])
        grouped = df.groupby("transaction_id")[amount_col].sum().reset_index()
        grouped.columns = ["transaction_id", out_col]
        return grouped

    fees_sum = sum_by_txn(tables.get("fees", pd.DataFrame()), "fee_amount", "total_fee")
    taxes_sum = sum_by_txn(tables.get("taxes", pd.DataFrame()), "tax_amount", "total_tax")
    refunds_sum = sum_by_txn(tables.get("refunds", pd.DataFrame()), "refund_amount", "total_refund")
    adjustments_sum = sum_by_txn(tables.get("adjustments", pd.DataFrame()), "adjustment_amount", "total_adjustment")

    # also track counts, since "how many adjustments" matters for anomaly
    # detection later (e.g. repeated_manual_adjustments)
    def count_by_txn(df, out_col):
        if df.empty or "transaction_id" not in df.columns:
            return pd.DataFrame(columns=["transaction_id", out_col])
        grouped = df.groupby("transaction_id").size().reset_index(name=out_col)
        return grouped

    adjustment_counts = count_by_txn(tables.get("adjustments", pd.DataFrame()), "adjustment_count")
    refund_counts = count_by_txn(tables.get("refunds", pd.DataFrame()), "refund_count")
    settlement_counts = count_by_txn(tables.get("settlements", pd.DataFrame()), "settlement_count")

    return fees_sum, taxes_sum, refunds_sum, adjustments_sum, adjustment_counts, refund_counts, settlement_counts


def build_reconciliation_table(tables):
    """
    Builds the master reconciliation DataFrame: one row per payment,
    joined with aggregated fees/taxes/refunds/adjustments and the
    settlement record (first settlement if duplicates exist -- the
    duplicates themselves are flagged via settlement_count).
    """
    payments = tables["payments"].copy()
    settlements = tables.get("settlements", pd.DataFrame()).copy()

    (fees_sum, taxes_sum, refunds_sum, adjustments_sum,
     adjustment_counts, refund_counts, settlement_counts) = aggregate_adjustments(tables)

    # Use the earliest settlement per transaction as "the" settlement record
    # for amount comparison; duplicate_settlement is flagged separately via
    # settlement_count, so we don't want duplicates to distort the amount check.
    if not settlements.empty:
        settlements_sorted = settlements.sort_values("settlement_timestamp")
        first_settlement = settlements_sorted.drop_duplicates(subset="transaction_id", keep="first")
    else:
        first_settlement = pd.DataFrame(columns=["transaction_id", "settlement_amount", "settlement_timestamp", "settlement_status"])

    recon = payments.merge(
        first_settlement[["transaction_id", "settlement_amount", "settlement_timestamp", "settlement_status"]],
        on="transaction_id", how="left"
    )
    recon = recon.merge(fees_sum, on="transaction_id", how="left")
    recon = recon.merge(taxes_sum, on="transaction_id", how="left")
    recon = recon.merge(refunds_sum, on="transaction_id", how="left")
    recon = recon.merge(adjustments_sum, on="transaction_id", how="left")
    recon = recon.merge(adjustment_counts, on="transaction_id", how="left")
    recon = recon.merge(refund_counts, on="transaction_id", how="left")
    recon = recon.merge(settlement_counts, on="transaction_id", how="left")

    fill_zero_cols = ["total_fee", "total_tax", "total_refund", "total_adjustment",
                       "adjustment_count", "refund_count", "settlement_count"]
    for col in fill_zero_cols:
        recon[col] = recon[col].fillna(0)

    # ---- the core financial formula ----
    # expected_settlement = payment - fees - taxes - refunds - adjustments
    recon["expected_settlement"] = (
        recon["amount"] - recon["total_fee"] - recon["total_tax"]
        - recon["total_refund"] - recon["total_adjustment"]
    ).round(2)

    recon["has_settlement"] = recon["settlement_amount"].notna()
    recon["actual_settlement"] = recon["settlement_amount"]

    recon["difference"] = (recon["actual_settlement"] - recon["expected_settlement"]).round(2)
    # when there's no settlement at all, difference is undefined -- handled
    # explicitly as "missing" in exceptions.py rather than treated as 0
    recon.loc[~recon["has_settlement"], "difference"] = None

    return recon


def main():
    from data_validator import validate_all

    tables, report = validate_all()
    if not report.is_valid:
        print(report.summary())
        print("\nAborting reconciliation -- fix validation errors first.")
        return

    recon = build_reconciliation_table(tables)
    print(f"Reconciliation table built: {len(recon)} rows")
    print(recon[["transaction_id", "amount", "expected_settlement", "actual_settlement", "difference"]].head(10))

    import os
    processed_dir = cfg.PROCESSED_DIR
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "reconciliation.csv")
    recon.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

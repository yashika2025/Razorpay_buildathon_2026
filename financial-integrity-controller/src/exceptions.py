"""
exceptions.py
-------------
Takes the reconciliation table produced by reconciliation.py and classifies
every transaction into one of a fixed set of statuses:

    MATCHED                    -- expected == actual, no adjustments needed
    RECONCILED_WITH_ADJUSTMENT -- expected == actual, but only after fees/
                                   taxes/refunds/manual adjustments were
                                   applied (this is a NORMAL, healthy case,
                                   not a problem)
    DUPLICATE_SETTLEMENT       -- more than one settlement record exists
                                   for this transaction
    MISSING_SETTLEMENT         -- no settlement record exists at all
    PARTIALLY_EXPLAINED        -- some but not all of the difference is
                                   explained by known adjustments
    UNRESOLVED                 -- the difference cannot be explained by any
                                   recorded fee/tax/refund/adjustment at all

Only MATCHED and RECONCILED_WITH_ADJUSTMENT are "reconciled" in the
dashboard's headline numbers. Everything else is an "exception" that flows
into anomaly detection and risk scoring.

WHY A SEPARATE MODULE FROM reconciliation.py:
Reconciliation computes financial FACTS (expected vs actual). Classification
is a separate judgment call about what those facts MEAN. Keeping them apart
means we can change the classification thresholds/logic without touching the
arithmetic, and vice versa.

Run directly: python src/exceptions.py
"""

import os
import pandas as pd

import config as cfg

# A difference smaller than this (in absolute rupees) is treated as fully
# explained -- protects against floating-point/rounding noise, not a
# business threshold.
FULL_MATCH_TOLERANCE = cfg.FULL_MATCH_TOLERANCE

# If the recorded adjustments (fee+tax+refund+adjustment) account for at
# least this fraction of the total gap between raw payment amount and
# actual settlement, we call it "partially explained" rather than fully
# "unresolved" -- there's a plausible partial story, just not a complete one.
PARTIAL_EXPLANATION_THRESHOLD = cfg.PARTIAL_EXPLANATION_THRESHOLD


def classify_row(row):
    """
    Returns (status, explanation_text) for a single reconciliation row.
    This function is intentionally simple, sequential if/elif logic --
    a finance controller needs to be able to read this and know exactly
    why a transaction landed in a given bucket.
    """
    if row["settlement_count"] > 1:
        return (
            "DUPLICATE_SETTLEMENT",
            f"{int(row['settlement_count'])} settlement records exist for this transaction; "
            f"only one is expected."
        )

    if not row["has_settlement"]:
        return (
            "MISSING_SETTLEMENT",
            "No settlement record was found for this payment."
        )

    diff = row["difference"]
    abs_diff = abs(diff) if diff is not None else None

    if abs_diff is not None and abs_diff <= FULL_MATCH_TOLERANCE:
        has_adjustments = (row["total_fee"] > 0 or row["total_tax"] > 0
                            or row["total_refund"] > 0 or row["total_adjustment"] > 0)
        if has_adjustments:
            parts = []
            if row["total_fee"] > 0:
                parts.append(f"fee \u20b9{row['total_fee']:.2f}")
            if row["total_tax"] > 0:
                parts.append(f"tax \u20b9{row['total_tax']:.2f}")
            if row["total_refund"] > 0:
                parts.append(f"refund \u20b9{row['total_refund']:.2f}")
            if row["total_adjustment"] > 0:
                parts.append(f"adjustment \u20b9{row['total_adjustment']:.2f}")
            return (
                "RECONCILED_WITH_ADJUSTMENT",
                f"Settlement matches expected amount after accounting for: {', '.join(parts)}."
            )
        else:
            return ("MATCHED", "Settlement amount exactly matches the payment amount.")

    # There's an unexplained gap. Figure out how much of the RAW difference
    # (payment amount vs actual settlement) is explained by recorded
    # adjustments, vs how much is genuinely unexplained.
    raw_gap = row["amount"] - row["actual_settlement"]
    recorded_adjustments_total = row["total_fee"] + row["total_tax"] + row["total_refund"] + row["total_adjustment"]

    if raw_gap == 0:
        explained_fraction = 0
    else:
        explained_fraction = min(recorded_adjustments_total / raw_gap, 1) if raw_gap != 0 else 0

    unexplained_amount = round(diff, 2) if diff is not None else None

    if 0 < explained_fraction < 1 and explained_fraction >= PARTIAL_EXPLANATION_THRESHOLD:
        return (
            "PARTIALLY_EXPLAINED",
            f"\u20b9{recorded_adjustments_total:.2f} of the \u20b9{raw_gap:.2f} gap is explained by "
            f"recorded fees/taxes/refunds/adjustments, but \u20b9{abs(unexplained_amount):.2f} remains unexplained."
        )

    return (
        "UNRESOLVED",
        f"\u20b9{abs(unexplained_amount):.2f} of the settlement difference cannot be explained "
        f"by any recorded fee, tax, refund, or adjustment."
    )


def classify_all(recon_df):
    """Applies classify_row to every transaction and appends the results
    as two new columns: status and explanation."""
    results = recon_df.apply(classify_row, axis=1, result_type="expand")
    results.columns = ["recon_status", "explanation"]
    out = pd.concat([recon_df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
    return out


RECONCILED_STATUSES = cfg.RECONCILED_STATUSES
EXCEPTION_STATUSES = cfg.EXCEPTION_STATUSES


def summarize(classified_df):
    total = len(classified_df)
    reconciled = classified_df["recon_status"].isin(RECONCILED_STATUSES).sum()
    exceptions = classified_df["recon_status"].isin(EXCEPTION_STATUSES).sum()
    match_rate = round(100 * reconciled / total, 2) if total else 0.0

    print(f"Total transactions:   {total}")
    print(f"Reconciled:           {reconciled}  ({match_rate}%)")
    print(f"Exceptions:           {exceptions}")
    print("\nBreakdown by status:")
    print(classified_df["recon_status"].value_counts())


def main():
    recon_path = os.path.join(cfg.PROCESSED_DIR, "reconciliation.csv")

    if not os.path.exists(recon_path):
        print("reconciliation.csv not found -- run reconciliation.py first.")
        return

    recon_df = pd.read_csv(recon_path)
    # has_settlement gets read back as string "True"/"False" from CSV; fix that
    recon_df["has_settlement"] = recon_df["has_settlement"].astype(bool)

    classified = classify_all(recon_df)
    summarize(classified)

    out_path = os.path.join(cfg.PROCESSED_DIR, "exceptions.csv")
    classified.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

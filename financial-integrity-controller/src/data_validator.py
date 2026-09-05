"""
data_validator.py
------------------
Validates the raw CSV files before they are fed into the reconciliation
engine. This catches structural problems (missing columns, bad types,
orphaned foreign keys) early, so that downstream components can assume
clean inputs and focus on financial logic instead of defensive parsing.

WHY THIS EXISTS AS ITS OWN MODULE:
In a real finance-controller pipeline, "garbage in, garbage out" is a real
risk -- a malformed settlement file could silently produce wrong exception
counts. Separating validation makes the pipeline auditable: we can point to
a validation report and say "we know the input data was structurally sound
before we ran any financial logic on it."

Run directly: python src/data_validator.py
"""

import os
import pandas as pd

import config as cfg

RAW_DIR = cfg.RAW_DIR

# Minimum required columns for each file. We don't enforce every optional
# column here -- only the ones the rest of the pipeline depends on.
REQUIRED_COLUMNS = {
    "payments": ["transaction_id", "merchant_id", "amount", "currency", "timestamp", "status"],
    "settlements": ["settlement_id", "transaction_id", "settlement_amount", "settlement_timestamp", "settlement_status"],
    "refunds": ["refund_id", "transaction_id", "refund_amount", "refund_timestamp"],
    "fees": ["fee_id", "transaction_id", "fee_amount"],
    "taxes": ["tax_id", "transaction_id", "tax_amount"],
    "adjustments": ["adjustment_id", "transaction_id", "adjustment_amount", "operator_id", "timestamp"],
    "audit_logs": ["log_id", "entity_type", "entity_id", "operator_id", "action", "timestamp"],
    "merchants": ["merchant_id", "name"],
}


class ValidationReport:
    """Collects issues found during validation so they can be printed or
    displayed in the dashboard, instead of failing silently or crashing."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def add_error(self, msg):
        self.errors.append(msg)

    def add_warning(self, msg):
        self.warnings.append(msg)

    @property
    def is_valid(self):
        # Warnings are tolerated (e.g. an orphaned refund is itself an
        # exception worth surfacing, not a reason to halt the pipeline).
        return len(self.errors) == 0

    def summary(self):
        lines = [f"Validation {'PASSED' if self.is_valid else 'FAILED'}"]
        lines.append(f"  Errors:   {len(self.errors)}")
        lines.append(f"  Warnings: {len(self.warnings)}")
        for e in self.errors:
            lines.append(f"    [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"    [WARN]  {w}")
        return "\n".join(lines)


def load_raw_tables(raw_dir=RAW_DIR):
    """Loads every CSV in REQUIRED_COLUMNS into a dict of DataFrames."""
    tables = {}
    for name in REQUIRED_COLUMNS:
        path = os.path.join(raw_dir, f"{name}.csv")
        if os.path.exists(path):
            tables[name] = pd.read_csv(path)
        else:
            tables[name] = pd.DataFrame()
    return tables


def validate_schema(tables, report):
    """Checks that each table has the columns the rest of the pipeline
    depends on. Missing columns are hard errors -- we can't safely proceed."""
    for name, required_cols in REQUIRED_COLUMNS.items():
        df = tables.get(name, pd.DataFrame())
        # payments, settlements, and merchants must always have rows.
        # fees, taxes, refunds, adjustments, and audit_logs are legitimately
        # optional -- a transaction (or an entire uploaded batch) may simply
        # have none of a given type, which is not a data-quality error.
        always_required_nonempty = {"payments", "settlements", "merchants"}
        if df.empty and name in always_required_nonempty:
            # payments/settlements/fees/taxes/merchants should never be empty
            report.add_error(f"'{name}' table is empty or missing.")
            continue
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report.add_error(f"'{name}' is missing required columns: {missing_cols}")


def validate_nulls(tables, report):
    """Flags nulls in financially-critical columns."""
    critical_cols = {
        "payments": ["transaction_id", "amount"],
        "settlements": ["transaction_id"],  # settlement_amount can legitimately be null pre-settlement
        "fees": ["transaction_id", "fee_amount"],
        "taxes": ["transaction_id", "tax_amount"],
    }
    for name, cols in critical_cols.items():
        df = tables.get(name, pd.DataFrame())
        for col in cols:
            if col in df.columns:
                n_null = df[col].isna().sum()
                if n_null > 0:
                    report.add_warning(f"'{name}.{col}' has {n_null} null value(s).")


def validate_duplicates(tables, report):
    """Flags duplicate primary keys and duplicate settlements per transaction
    (the latter is a legitimate exception scenario, so it's a warning, not
    a hard error)."""
    pk_cols = {
        "payments": "transaction_id",
        "settlements": "settlement_id",
        "refunds": "refund_id",
        "fees": "fee_id",
        "taxes": "tax_id",
        "adjustments": "adjustment_id",
        "audit_logs": "log_id",
        "merchants": "merchant_id",
    }
    for name, pk in pk_cols.items():
        df = tables.get(name, pd.DataFrame())
        if pk in df.columns:
            n_dupes = df[pk].duplicated().sum()
            if n_dupes > 0:
                report.add_error(f"'{name}' has {n_dupes} duplicate primary key ('{pk}') value(s).")

    settlements = tables.get("settlements", pd.DataFrame())
    if not settlements.empty and "transaction_id" in settlements.columns:
        dupe_txns = settlements["transaction_id"].value_counts()
        dupe_txns = dupe_txns[dupe_txns > 1]
        if len(dupe_txns) > 0:
            report.add_warning(
                f"{len(dupe_txns)} transaction(s) have more than one settlement record "
                f"(this is expected to appear later as 'duplicate_settlement' exceptions)."
            )


def validate_referential_integrity(tables, report):
    """Checks that foreign keys (transaction_id, merchant_id) point to real
    records. Orphaned records are financially meaningful (e.g. a refund with
    no matching payment is itself an integrity problem) so these are
    warnings that get surfaced downstream, not silent failures."""
    payments = tables.get("payments", pd.DataFrame())
    valid_txn_ids = set(payments["transaction_id"]) if "transaction_id" in payments.columns else set()

    merchants = tables.get("merchants", pd.DataFrame())
    valid_merchant_ids = set(merchants["merchant_id"]) if "merchant_id" in merchants.columns else set()

    for name in ["settlements", "refunds", "fees", "taxes", "adjustments"]:
        df = tables.get(name, pd.DataFrame())
        if "transaction_id" in df.columns and len(df) > 0:
            orphaned = ~df["transaction_id"].isin(valid_txn_ids)
            n_orphaned = orphaned.sum()
            if n_orphaned > 0:
                report.add_warning(
                    f"'{name}' has {n_orphaned} record(s) referencing a transaction_id "
                    f"not found in payments."
                )

    if "merchant_id" in payments.columns and len(payments) > 0:
        orphaned_merchants = ~payments["merchant_id"].isin(valid_merchant_ids)
        n_orphaned_merchants = orphaned_merchants.sum()
        if n_orphaned_merchants > 0:
            report.add_warning(
                f"'payments' has {n_orphaned_merchants} record(s) referencing an "
                f"unknown merchant_id."
            )


def validate_amounts(tables, report):
    """Sanity-checks that monetary fields are non-negative where they
    should always be (a negative fee/tax would indicate a data problem,
    not a financial exception)."""
    non_negative_checks = {
        "payments": "amount",
        "fees": "fee_amount",
        "taxes": "tax_amount",
        "refunds": "refund_amount",
    }
    for name, col in non_negative_checks.items():
        df = tables.get(name, pd.DataFrame())
        if col in df.columns and len(df) > 0:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                report.add_error(f"'{name}.{col}' has {n_negative} negative value(s).")


def validate_all(raw_dir=RAW_DIR):
    """Runs the full validation suite and returns (tables, report)."""
    tables = load_raw_tables(raw_dir)
    report = ValidationReport()
    validate_schema(tables, report)
    validate_nulls(tables, report)
    validate_duplicates(tables, report)
    validate_referential_integrity(tables, report)
    validate_amounts(tables, report)
    return tables, report


def main():
    tables, report = validate_all()
    print(report.summary())
    if not report.is_valid:
        print("\nValidation failed -- fix errors above before running reconciliation.")
    else:
        print("\nData is structurally sound. Safe to proceed to reconciliation.")


if __name__ == "__main__":
    main()

"""
data_generator.py
------------------
Generates a realistic, synthetic multi-source financial dataset for the
AI Financial Integrity & Risk Controller.

WHY THIS EXISTS:
Every downstream component (reconciliation, anomaly detection, risk scoring)
needs data that behaves like a real payments pipeline: most transactions are
clean, a smaller set has legitimate adjustments (fees/taxes/refunds), a
smaller set still has operational errors, and a very small set has
suspicious/abnormal patterns worth investigating.

We generate transactions in labeled "scenario buckets" so that we always know
the ground truth of WHY a transaction was built the way it was. This ground
truth is saved to data/raw/scenario_labels.csv and is used ONLY for our own
evaluation later (Section 15 of the plan) -- it is never given to the
reconciliation or anomaly detection engines, since in real life you would
not know this in advance.

Run directly: python src/data_generator.py
"""

import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as cfg

# ----------------------------------------------------------------------
# CONFIG -- pulled from config.py. See that file to change any of these.
# ----------------------------------------------------------------------
RANDOM_SEED = cfg.RANDOM_SEED
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_MERCHANTS = cfg.NUM_MERCHANTS
NUM_TRANSACTIONS = cfg.NUM_TRANSACTIONS
START_DATE = datetime(*cfg.DATASET_START_DATE)
END_DATE = datetime(*cfg.DATASET_END_DATE)

PAYMENT_METHODS = cfg.PAYMENT_METHODS
CURRENCY = cfg.CURRENCY

OUTPUT_DIR = cfg.RAW_DIR

# Scenario buckets and their approximate share of all transactions.
# These proportions are deliberately designed so that:
#   - most transactions are boring and reconcile cleanly
#   - a meaningful minority have legitimate financial adjustments
#   - a small minority are operational errors
#   - a very small minority are security/integrity anomalies
# This keeps anomaly detection non-trivial (rare, not half the dataset).
SCENARIO_WEIGHTS = cfg.SCENARIO_WEIGHTS

assert abs(sum(SCENARIO_WEIGHTS.values()) - 1.0) < 1e-6, "Scenario weights must sum to 1"


def random_timestamp(start=START_DATE, end=END_DATE):
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def business_hours_timestamp(start=START_DATE, end=END_DATE):
    """A timestamp that falls within normal 9am-9pm operating hours."""
    ts = random_timestamp(start, end)
    hour = random.randint(9, 20)
    return ts.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def odd_hours_timestamp(start=START_DATE, end=END_DATE):
    """A timestamp between 1am-4am, used for suspicious activity scenarios."""
    ts = random_timestamp(start, end)
    hour = random.choice([1, 2, 3, 4])
    return ts.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def new_id(prefix):
    return f"{prefix}{uuid.uuid4().hex[:8].upper()}"


# ----------------------------------------------------------------------
# MERCHANTS
# ----------------------------------------------------------------------
def generate_merchants(n=NUM_MERCHANTS):
    categories = ["ecommerce", "food_delivery", "travel", "saas", "retail", "utilities"]
    merchants = []
    for i in range(n):
        merchants.append({
            "merchant_id": f"MERCH{i+1:04d}",
            "name": f"Merchant_{i+1:04d}",
            "category": random.choice(categories),
            "onboarding_date": (START_DATE - timedelta(days=random.randint(30, 900))).date().isoformat(),
            "historical_avg_daily_txns": random.randint(5, 200),
        })
    return pd.DataFrame(merchants)


# ----------------------------------------------------------------------
# A small pool of "operators" (finance-ops staff who can create manual
# adjustments / settlements). Most operators behave normally; a couple
# are deliberately given abnormal behavior for the security scenarios.
# ----------------------------------------------------------------------
OPERATORS = [f"OP{i:03d}" for i in range(1, cfg.NUM_OPERATORS + 1)]
SUSPICIOUS_OPERATORS = cfg.SUSPICIOUS_OPERATORS

DEVICES = [f"DEV{i:03d}" for i in range(1, cfg.NUM_DEVICES + 1)]
SUSPICIOUS_DEVICE = cfg.SUSPICIOUS_DEVICE

IPS = [f"10.0.{random.randint(0,255)}.{random.randint(0,255)}" for _ in range(20)]
SUSPICIOUS_IP = cfg.SUSPICIOUS_IP


def pick_scenario():
    scenarios, weights = zip(*SCENARIO_WEIGHTS.items())
    return random.choices(scenarios, weights=weights, k=1)[0]


# ----------------------------------------------------------------------
# MAIN GENERATION LOOP
# Builds one transaction at a time according to its assigned scenario,
# appending rows to each of the record-type tables as appropriate.
# ----------------------------------------------------------------------
def generate_dataset(n=NUM_TRANSACTIONS, merchants_df=None):
    merchant_ids = merchants_df["merchant_id"].tolist()

    payments, settlements, refunds, fees, taxes, adjustments, audit_logs = (
        [], [], [], [], [], [], []
    )
    scenario_labels = []

    for i in range(n):
        scenario = pick_scenario()
        txn_id = f"TXN{i+1:06d}"
        merchant_id = random.choice(merchant_ids)
        customer_id = new_id("CUST")
        base_amount = round(random.uniform(200, 25000), 2)
        payment_ts = business_hours_timestamp()

        payments.append({
            "transaction_id": txn_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "amount": base_amount,
            "currency": CURRENCY,
            "timestamp": payment_ts.isoformat(),
            "status": "SUCCESS",
            "payment_method": random.choice(PAYMENT_METHODS),
        })

        fee_amount = round(base_amount * random.uniform(*cfg.FEE_RATE_RANGE), 2)
        tax_amount = round(fee_amount * cfg.GST_ON_FEE_RATE, 2)
        refund_amount = 0.0
        adjustment_amount = 0.0
        settlement_amount = base_amount
        settlement_status = "SETTLED"
        settlement_ts = payment_ts + timedelta(days=random.choice([1, 2]))

        # ---------------- scenario-specific construction ----------------
        if scenario == "exact_match":
            fee_amount = 0.0
            tax_amount = 0.0
            settlement_amount = base_amount

        elif scenario == "fee_only":
            settlement_amount = round(base_amount - fee_amount, 2)
            tax_amount = 0.0

        elif scenario == "tax_only":
            fee_amount = 0.0
            settlement_amount = round(base_amount - tax_amount, 2)

        elif scenario == "fee_and_tax":
            settlement_amount = round(base_amount - fee_amount - tax_amount, 2)

        elif scenario == "legitimate_refund":
            refund_amount = round(base_amount * random.uniform(0.1, 1.0), 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - refund_amount, 2)
            refunds.append({
                "refund_id": new_id("RFD"),
                "transaction_id": txn_id,
                "refund_amount": refund_amount,
                "refund_timestamp": (settlement_ts + timedelta(hours=random.randint(1, 48))).isoformat(),
                "reason": random.choice(["customer_request", "order_cancelled", "product_return"]),
            })

        elif scenario == "normal_delay":
            settlement_amount = round(base_amount - fee_amount - tax_amount, 2)
            settlement_ts = payment_ts + timedelta(days=random.randint(3, 6))  # longer but still normal

        elif scenario == "missing_settlement":
            settlement_amount = None  # no settlement row will be created

        elif scenario == "duplicate_settlement":
            settlement_amount = round(base_amount - fee_amount - tax_amount, 2)
            # a second, duplicate settlement row will be appended below

        elif scenario == "incorrect_amount":
            # operational error: settlement amount doesn't match expected net
            # by a random, non-adjustment-explained amount
            error = round(random.uniform(50, 2000), 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - error, 2)

        elif scenario == "duplicate_refund":
            refund_amount = round(base_amount * random.uniform(0.1, 0.5), 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - refund_amount, 2)
            for _ in range(2):  # duplicated refund record
                refunds.append({
                    "refund_id": new_id("RFD"),
                    "transaction_id": txn_id,
                    "refund_amount": refund_amount,
                    "refund_timestamp": (settlement_ts + timedelta(hours=random.randint(1, 48))).isoformat(),
                    "reason": "customer_request",
                })

        elif scenario == "missing_refund_record":
            # money was actually refunded (settlement is lower) but no refund
            # record exists to explain it -> becomes an unexplained exception
            refund_amount = round(base_amount * random.uniform(0.1, 0.5), 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - refund_amount, 2)
            # deliberately NOT appending to refunds list

        elif scenario == "incorrect_fee":
            wrong_fee = round(fee_amount + random.uniform(100, 800), 2)
            settlement_amount = round(base_amount - wrong_fee - tax_amount, 2)
            fee_amount = wrong_fee  # fee record itself is wrong

        elif scenario == "unusual_amount_modification":
            adjustment_amount = round(random.uniform(1000, 8000), 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - adjustment_amount, 2)
            op = random.choice(SUSPICIOUS_OPERATORS)
            adj_ts = odd_hours_timestamp()
            adjustments.append({
                "adjustment_id": new_id("ADJ"),
                "transaction_id": txn_id,
                "adjustment_amount": adjustment_amount,
                "reason": "manual_correction",
                "operator_id": op,
                "timestamp": adj_ts.isoformat(),
            })
            audit_logs.append({
                "log_id": new_id("LOG"),
                "entity_type": "transaction",
                "entity_id": txn_id,
                "operator_id": op,
                "action": "amount_modified",
                "previous_value": str(base_amount),
                "new_value": str(round(base_amount - adjustment_amount, 2)),
                "timestamp": adj_ts.isoformat(),
                "device_id": SUSPICIOUS_DEVICE,
                "ip_address": SUSPICIOUS_IP,
            })

        elif scenario == "repeated_manual_adjustments":
            op = random.choice(SUSPICIOUS_OPERATORS)
            base_ts = odd_hours_timestamp()
            total_adj = 0.0
            for k in range(random.randint(4, 8)):  # many small adjustments, short window
                amt = round(random.uniform(200, 1500), 2)
                total_adj += amt
                ts_k = base_ts + timedelta(minutes=k * random.randint(1, 5))
                adjustments.append({
                    "adjustment_id": new_id("ADJ"),
                    "transaction_id": txn_id,
                    "adjustment_amount": amt,
                    "reason": "manual_correction",
                    "operator_id": op,
                    "timestamp": ts_k.isoformat(),
                })
                audit_logs.append({
                    "log_id": new_id("LOG"),
                    "entity_type": "transaction",
                    "entity_id": txn_id,
                    "operator_id": op,
                    "action": "amount_modified",
                    "previous_value": str(base_amount),
                    "new_value": str(round(base_amount - total_adj, 2)),
                    "timestamp": ts_k.isoformat(),
                    "device_id": SUSPICIOUS_DEVICE,
                    "ip_address": SUSPICIOUS_IP,
                })
            adjustment_amount = round(total_adj, 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - adjustment_amount, 2)

        elif scenario == "suspicious_operator_activity":
            # one operator touching an unusually large number of unrelated
            # transactions is modeled at the audit_log level in bulk after
            # the main loop; here we just tag this transaction as touched
            op = random.choice(SUSPICIOUS_OPERATORS)
            adj_ts = business_hours_timestamp()
            adjustment_amount = round(random.uniform(100, 3000), 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - adjustment_amount, 2)
            adjustments.append({
                "adjustment_id": new_id("ADJ"),
                "transaction_id": txn_id,
                "adjustment_amount": adjustment_amount,
                "reason": "manual_correction",
                "operator_id": op,
                "timestamp": adj_ts.isoformat(),
            })
            audit_logs.append({
                "log_id": new_id("LOG"),
                "entity_type": "transaction",
                "entity_id": txn_id,
                "operator_id": op,
                "action": "amount_modified",
                "previous_value": str(base_amount),
                "new_value": str(settlement_amount),
                "timestamp": adj_ts.isoformat(),
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IPS),
            })

        elif scenario == "unusual_hours_activity":
            settlement_amount = round(base_amount - fee_amount - tax_amount, 2)
            adj_ts = odd_hours_timestamp()
            op = random.choice(OPERATORS)
            audit_logs.append({
                "log_id": new_id("LOG"),
                "entity_type": "settlement",
                "entity_id": txn_id,
                "operator_id": op,
                "action": "settlement_status_modified",
                "previous_value": "PENDING",
                "new_value": "SETTLED",
                "timestamp": adj_ts.isoformat(),
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IPS),
            })

        elif scenario == "high_adjustment_frequency":
            op = random.choice(SUSPICIOUS_OPERATORS)
            total_adj = 0.0
            base_ts = business_hours_timestamp()
            for k in range(random.randint(6, 10)):
                amt = round(random.uniform(50, 500), 2)
                total_adj += amt
                ts_k = base_ts + timedelta(minutes=k * 2)
                adjustments.append({
                    "adjustment_id": new_id("ADJ"),
                    "transaction_id": txn_id,
                    "adjustment_amount": amt,
                    "reason": "batch_correction",
                    "operator_id": op,
                    "timestamp": ts_k.isoformat(),
                })
            adjustment_amount = round(total_adj, 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - adjustment_amount, 2)

        elif scenario == "suspicious_device_ip_correlation":
            op = random.choice(OPERATORS)
            adj_ts = business_hours_timestamp()
            adjustment_amount = round(random.uniform(500, 4000), 2)
            settlement_amount = round(base_amount - fee_amount - tax_amount - adjustment_amount, 2)
            adjustments.append({
                "adjustment_id": new_id("ADJ"),
                "transaction_id": txn_id,
                "adjustment_amount": adjustment_amount,
                "reason": "manual_correction",
                "operator_id": op,
                "timestamp": adj_ts.isoformat(),
            })
            # same suspicious device+IP pair reused across unrelated operators/txns
            audit_logs.append({
                "log_id": new_id("LOG"),
                "entity_type": "transaction",
                "entity_id": txn_id,
                "operator_id": op,
                "action": "amount_modified",
                "previous_value": str(base_amount),
                "new_value": str(settlement_amount),
                "timestamp": adj_ts.isoformat(),
                "device_id": SUSPICIOUS_DEVICE,
                "ip_address": SUSPICIOUS_IP,
            })

        # ---------------- write settlement + fee + tax rows ----------------
        if scenario != "missing_settlement":
            settlements.append({
                "settlement_id": new_id("STL"),
                "transaction_id": txn_id,
                "settlement_amount": settlement_amount,
                "settlement_timestamp": settlement_ts.isoformat(),
                "settlement_status": settlement_status,
            })
            if scenario == "duplicate_settlement":
                settlements.append({
                    "settlement_id": new_id("STL"),
                    "transaction_id": txn_id,
                    "settlement_amount": settlement_amount,
                    "settlement_timestamp": (settlement_ts + timedelta(hours=2)).isoformat(),
                    "settlement_status": settlement_status,
                })

        if fee_amount > 0:
            fees.append({
                "fee_id": new_id("FEE"),
                "transaction_id": txn_id,
                "fee_amount": fee_amount,
                "fee_type": "gateway_processing_fee",
            })
        if tax_amount > 0:
            taxes.append({
                "tax_id": new_id("TAX"),
                "transaction_id": txn_id,
                "tax_amount": tax_amount,
                "tax_type": "GST",
            })

        scenario_labels.append({"transaction_id": txn_id, "scenario": scenario})

    dfs = {
        "payments": pd.DataFrame(payments),
        "settlements": pd.DataFrame(settlements),
        "refunds": pd.DataFrame(refunds),
        "fees": pd.DataFrame(fees),
        "taxes": pd.DataFrame(taxes),
        "adjustments": pd.DataFrame(adjustments),
        "audit_logs": pd.DataFrame(audit_logs),
        "scenario_labels": pd.DataFrame(scenario_labels),
    }
    return dfs


def inject_bulk_suspicious_operator_activity(dfs, n_extra_touches=cfg.BULK_SUSPICIOUS_TOUCHES):
    """
    Reinforces the 'suspicious_operator_activity' story: one operator
    (OP010) touches far more transactions than any normal operator would
    in the same window, simulating a compromised account or insider
    scenario. We add extra audit_log rows referencing EXISTING transactions
    so this shows up clearly as a behavioral outlier without needing new
    fake transactions.
    """
    txn_ids = dfs["payments"]["transaction_id"].sample(
        n=min(n_extra_touches, len(dfs["payments"])), random_state=RANDOM_SEED
    ).tolist()
    extra_logs = []
    burst_start = odd_hours_timestamp()
    for i, txn_id in enumerate(txn_ids):
        ts = burst_start + timedelta(minutes=i)
        extra_logs.append({
            "log_id": new_id("LOG"),
            "entity_type": "transaction",
            "entity_id": txn_id,
            "operator_id": "OP010",
            "action": "record_viewed_and_modified",
            "previous_value": "N/A",
            "new_value": "N/A",
            "timestamp": ts.isoformat(),
            "device_id": SUSPICIOUS_DEVICE,
            "ip_address": SUSPICIOUS_IP,
        })
    dfs["audit_logs"] = pd.concat(
        [dfs["audit_logs"], pd.DataFrame(extra_logs)], ignore_index=True
    )
    return dfs


def save_dataset(dfs, output_dir=OUTPUT_DIR):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for name, df in dfs.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(df)} rows)")


def main():
    print("Generating merchants...")
    merchants_df = generate_merchants()

    print(f"Generating {NUM_TRANSACTIONS} transactions across scenario buckets...")
    dfs = generate_dataset(NUM_TRANSACTIONS, merchants_df)
    dfs["merchants"] = merchants_df

    print("Injecting bulk suspicious operator behavior (OP010)...")
    dfs = inject_bulk_suspicious_operator_activity(dfs)

    print("Saving CSV files to data/raw/ ...")
    save_dataset(dfs)

    print("\nScenario distribution:")
    print(dfs["scenario_labels"]["scenario"].value_counts())


if __name__ == "__main__":
    main()

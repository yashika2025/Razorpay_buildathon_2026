# AI Financial Integrity & Risk Controller

**Razorpay AI Buildathon 2026 — AI Finance Controller track**

An AI-assisted financial operations controller that reconciles payment records across multiple sources, explains discrepancies, and layers on a financial-integrity/anomaly-detection module to flag unusual patterns for human investigation — with every score fully explainable.

---

## 1. Problem Statement

Payment platforms process settlements across multiple internal systems — the payment gateway, the settlement engine, fee/tax calculation, refund processing, and manual adjustments. The amount a merchant actually receives rarely equals the raw payment amount, and figuring out *why* is a constant finance-operations burden. Some gaps are legitimate (fees, taxes, refunds, processing delays); some are operational errors (duplicate records, wrong amounts); and a small number reflect abnormal or potentially manipulated activity that deserves security review. Finance teams need a system that distinguishes these cases automatically, explains its reasoning, and prioritizes what actually needs a human to look at it.

## 2. Why This Matters

Reconciliation failures that go unexplained are both a finance problem (money that can't be accounted for) and a security problem (an unexplained gap is exactly what a manipulated record or compromised account would look like). Most tooling treats these as separate concerns — an accounting reconciliation tool on one side, a fraud-detection model on the other. This project treats them as one pipeline: financial facts first, security judgment second, always traceable back to the same underlying data.

## 3. Razorpay Finance Controller Track Alignment

This directly implements the core ask of the track: reconcile financial records across multiple sources, identify discrepancies, explain legitimate differences, and surface unresolved exceptions. It extends that with a security/integrity layer relevant to a payments company, where settlement integrity is a security concern as much as an accounting one — without ever overstepping into an unsubstantiated fraud claim.

## 4. Solution

A deterministic reconciliation engine computes financial facts (expected vs. actual settlement, adjustment-aware). A separate classification layer turns those facts into a fixed set of exception statuses. A behavioral anomaly detector (rule-based + Isolation Forest) and an entity-correlation module (operator/device/IP, via NetworkX) analyze the same data from a security angle — independently of whether a transaction is financially resolved. A transparent, additive risk scorer combines both views into one explained score per transaction. A Streamlit dashboard presents all of this, with an optional LLM assistant that explains — never invents — the computed facts.

## 5. Architecture

```
Data ingestion (8 synthetic CSV sources)
        |
Data validation (schema, nulls, referential integrity)
        |
Reconciliation engine (expected vs. actual, adjustment-aware)
        |
Exception classification (6 statuses)
        |
        +---------------------+
        |                     |
Anomaly detection      Security analysis
(rules + Isolation     (operator/device/IP
 Forest, per-txn         entity correlation,
 behavior)                NetworkX graph)
        |                     |
        +---------------------+
                  |
          Risk scoring (additive, explained)
                  |
          Streamlit dashboard
                  |
      Optional LLM investigation assistant
```

Reconciliation and classification are pure, deterministic financial logic — no ML. Anomaly detection and security analysis run in parallel because a transaction can be financially self-consistent (correctly recorded adjustment) while still being behaviorally suspicious, or vice versa. Risk scoring is the only place these two views are combined, and it does so additively so every point on the score is traceable to a specific signal.

## 6. Features

**Finance Controller (core)**
- Synthetic multi-source dataset generator (3,000 transactions, 18 labeled scenario types)
- Data validation layer (schema, null, duplicate, referential-integrity, amount-sanity checks)
- Adjustment-aware reconciliation engine (fees, taxes, refunds, manual adjustments)
- Exception classification into 6 statuses
- Reconciliation dashboard view with filtering

**Security / Integrity layer**
- Rule-based behavioral anomaly detection (adjustment frequency, odd-hours activity, rapid bursts, large single adjustments, operator activity outliers)
- Isolation Forest anomaly detection over behavioral features (unsupervised)
- Entity correlation graph (operator/device/IP) via NetworkX, with MAD-based robust outlier detection
- Transparent, additive risk scoring (LOW / MEDIUM / HIGH) with itemized reasons
- Investigation queue in the dashboard

**Optional**
- LLM investigation assistant (explains pre-computed facts only, never invents numbers)

**Evaluation**
- Reconciliation and anomaly metrics computed directly from the generated dataset
- Full test suite covering all required scenarios

## 7. Dataset

Generated by `src/data_generator.py` into `data/raw/`:

| File | Description |
|---|---|
| `merchants.csv` | Merchant identity and category |
| `payments.csv` | The originating payment transaction |
| `settlements.csv` | Settlement record(s) per transaction |
| `refunds.csv` | Refund records |
| `fees.csv` | Gateway processing fees |
| `taxes.csv` | GST on fees |
| `adjustments.csv` | Manual adjustments, with operator and timestamp |
| `audit_logs.csv` | Operator actions with device/IP and previous/new values |
| `scenario_labels.csv` | Ground-truth scenario per transaction — used **only** for our own evaluation, never given to the pipeline |

3,000 transactions are split across 18 scenario buckets: ~56% exact matches, ~31% legitimate fee/tax/refund/delay cases, ~7% operational errors (missing/duplicate settlements, wrong amounts, duplicate/missing refunds, wrong fees), and ~2% security/integrity anomalies (unusual amount modifications, repeated manual adjustments, suspicious operator activity, odd-hours activity, high adjustment frequency, shared device/IP). The proportions are deliberately skewed toward "normal" so that anomaly detection is a real problem, not a trivial one.

## 8. Reconciliation Methodology

For every payment, `expected_settlement = amount - total_fee - total_tax - total_refund - total_adjustment`, where each total is summed from however many fee/tax/refund/adjustment records exist for that transaction (zero, one, or many). `actual_settlement` is read from the first settlement record for that transaction. The difference between the two, plus separate checks for missing/duplicate settlement records, drives the classification:

- **MATCHED** — no adjustments needed, amounts equal
- **RECONCILED_WITH_ADJUSTMENT** — amounts equal once known fees/taxes/refunds/adjustments are accounted for
- **PARTIALLY_EXPLAINED** — known adjustments explain most, but not all, of the gap
- **UNRESOLVED** — no recorded adjustment explains the gap
- **MISSING_SETTLEMENT** / **DUPLICATE_SETTLEMENT** — structural issues, checked before amount comparison

Only MATCHED and RECONCILED_WITH_ADJUSTMENT count as "reconciled" for the headline match rate.

## 9. Anomaly Detection Methodology

Runs on behavioral features derived from `adjustments.csv` and `audit_logs.csv`, for any transaction with adjustment/audit activity — regardless of its reconciliation status. This is deliberate: a manipulated transaction can be financially self-consistent (the adjustment record honestly explains the settlement gap) while still being behaviorally abnormal.

**Why Isolation Forest**: there is no reliable ground-truth fraud label in a real deployment, so a supervised classifier isn't appropriate. Isolation Forest is designed to isolate rare, different-looking points without labels, which matches this problem well.

**Features used**: adjustment count, total/max adjustment amount, fraction of payment amount in the largest single adjustment, count of odd-hours (midnight–6am) adjustments, distinct operators involved, time span between adjustments, operator's total transactions touched, and that operator's activity z-score.

**What the output means**: a rule-based flag means a specific, named, human-readable threshold was crossed (e.g. "4+ adjustments on one transaction"). An Isolation Forest outlier means the model found this transaction's overall feature combination unusual relative to the rest of the dataset, without any single feature necessarily crossing a fixed threshold.

**Limitations**: this is unsupervised and evaluated against our own synthetic ground truth, not real-world fraud outcomes. Isolation Forest on a small, synthetic dataset can overfit to the exact patterns we injected — a caveat we state plainly rather than claim classification accuracy we haven't earned. See `data/processed/anomaly_features.csv` for the raw feature table.

## 10. Cybersecurity Layer

**The security story**: financial systems depend on data integrity. An attacker, malicious insider, compromised account, or erroneous process could manipulate financial records or generate abnormal financial activity. This system monitors financial integrity, detects abnormal behavior, correlates activity across operator/device/IP, assigns risk, prioritizes investigation, and provides evidence for human review — it does not attempt to be a conventional penetration-testing tool, and it never claims to have detected confirmed fraud.

**Entity correlation**: `src/security_analysis.py` builds a NetworkX graph connecting operators, devices, and IP addresses via the audit log, then computes:
- Operator activity outliers, using a **MAD-based robust z-score** rather than a standard mean/std z-score. This mattered in practice — during development, a standard z-score masked one of two abnormally active operators because the other operator's extreme activity inflated the standard deviation used in the calculation. Median absolute deviation is far less sensitive to a small number of extreme points, so both surfaced correctly once we switched.
- Devices/IPs shared across multiple operators, which can indicate a compromised endpoint or shared credentials.

## 11. Risk Scoring

An additive score out of 100, built from four capped components so no single signal can dominate unfairly:

| Component | Max points | What it measures |
|---|---|---|
| Financial discrepancy severity | 30 | Size of the unexplained gap relative to the payment amount, or a missing/duplicate settlement |
| Behavioral anomaly (rule-based) | 30 | Count of specific, named rule violations (frequency, odd hours, burst timing, large single adjustment, operator outlier) |
| ML anomaly score | 20 | Isolation Forest's anomaly score for this transaction |
| Entity correlation | 20 | Operator/device/IP flags from the security analysis graph |

Risk levels: **LOW** (0–29), **MEDIUM** (30–59), **HIGH** (60–100). A transaction with zero financial discrepancy can still reach HIGH purely on behavioral + ML + entity signals — intentional, since a compromised operator who correctly records their own adjustment leaves the books balanced, but the pattern of how and when they did it is still the strongest signal available.

Every score ships with the specific, itemized reasons behind it (e.g. *"8 manual adjustments on a single transaction | 8 adjustment(s) made between midnight–6am | operator OP010 has touched 125 transactions, far above the typical operator baseline | IP 203.0.113.77 shared across multiple operators"*) — never a bare number.

## 12. Evaluation Metrics

Computed directly from the 3,000-transaction synthetic dataset (run the pipeline to regenerate these):

- **Reconciliation**: 2,801 / 3,000 reconciled (93.37% match rate), 199 exceptions
- **Exception breakdown**: 102 unresolved, 51 missing settlement, 40 duplicate settlement, 6 partially explained
- **Anomaly detection**: 94 transactions had adjustment/audit activity to analyze; 84 triggered at least one rule-based flag; 10 were flagged as Isolation Forest outliers
- **Risk scoring**: 4 HIGH-risk, 65 MEDIUM-risk, remainder LOW

No ground-truth fraud labels exist for this dataset (by design — see Anomaly Detection Methodology), so no precision/recall/F1 is reported for the anomaly model. `scenario_labels.csv` (our own injected ground truth, used only for internal validation) confirms the classifier and anomaly detector correctly separate the intended categories — see the crosstabs referenced in development notes.

## 13. Installation

```bash
git clone <your-repo-url>
cd financial-integrity-controller
pip install -r requirements.txt
```

**Configuration**: every tunable value in this project (dataset size, anomaly thresholds, risk-score weights, which LLM provider to use) lives in one place: `src/config.py`. You shouldn't need to hunt through other files to change how the system behaves.

**LLM provider (optional, only for the investigation assistant)**: pick one in `src/config.py` by setting `LLM_PROVIDER`:

| Provider | Cost | Setup |
|---|---|---|
| `"ollama"` (default) | $0 forever, no key | Install [Ollama](https://ollama.com), run `ollama serve`, then `ollama pull llama3.2`. Best for local demos; if you deploy the app online, Ollama needs to keep running on a machine you control. |
| `"groq"` | Free tier | Get a key at [console.groq.com](https://console.groq.com) |
| `"gemini"` | Free tier | Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `"anthropic"` | Small free trial, then paid | Get a key at [console.anthropic.com](https://console.anthropic.com) |

For any hosted option (everything except Ollama):
```bash
cp .env.example .env
# then edit .env and paste in the key for whichever provider you picked
```
`src/llm_assistant.py` loads `.env` automatically — nothing else in the project touches API keys.

## 14. Usage

Run the pipeline stages in order (or click "Run full pipeline now" in the dashboard sidebar):

```bash
python src/data_generator.py
python src/data_validator.py
python src/reconciliation.py
python src/exceptions.py
python src/anomaly_detection.py
python src/security_analysis.py
python src/risk_scoring.py
```

Then launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Run tests:

```bash
python -m pytest tests/ -v
```

The optional LLM investigation assistant activates automatically once your `.env` file has a valid key — no other setup needed.

## 15. Limitations

- All data is synthetic; anomaly detection results are illustrative of the approach, not validated against real fraud outcomes.
- Isolation Forest is fit on a relatively small, synthetic feature set and can overfit to the exact patterns we injected.
- Entity correlation is intentionally simple (pairwise counts over a small graph) — it is not a fraud-ring detection system.
- The LLM investigation assistant, when enabled, is only as good as the facts it's given; it cannot answer questions the underlying data doesn't support, and is explicitly instructed not to guess.
- Risk scoring thresholds and point allocations are reasoned and documented but ultimately calibrated choices, not derived from a labeled outcome dataset — they should be revisited against real operational data before any production use.

## 16. Future Improvements

- Replace synthetic data with a de-identified sample of real transaction data for realistic calibration.
- Add supervised learning once real investigation outcomes provide labels.
- Expand entity correlation into genuine graph-based ring detection if the operational need justifies the added complexity.
- Add a feedback loop where investigator decisions (confirmed vs. dismissed) retrain or recalibrate the risk score.

## 17. Technology Stack

Python, Pandas, NumPy, Scikit-learn (Isolation Forest), Streamlit, NetworkX, pytest. Optional: Anthropic API for the LLM investigation assistant.

## 18. Project Structure

```
financial-integrity-controller/
├── data/
│   ├── raw/            (generated CSV sources)
│   └── processed/      (reconciliation, exceptions, anomaly, risk outputs)
├── src/
│   ├── data_generator.py
│   ├── data_validator.py
│   ├── reconciliation.py
│   ├── exceptions.py
│   ├── anomaly_detection.py
│   ├── security_analysis.py
│   ├── risk_scoring.py
│   └── llm_assistant.py
├── dashboard/
│   └── app.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 19. What This Project Does Not Claim

This system does not claim to detect confirmed fraud, does not claim to outperform any existing financial platform, and does not claim validated accuracy against real-world outcomes. It surfaces statistically unusual, well-explained patterns for human investigators to review — that is the entire scope of the claim.

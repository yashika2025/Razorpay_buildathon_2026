"""
config.py
----------
EVERY tunable value in this project lives here, in one place, instead of
scattered across 7 different files. If you want to change how the system
behaves -- dataset size, anomaly thresholds, risk score weights, which LLM
model to use -- this is the only file you should need to touch.

Nothing in this file is a secret. Your API key does NOT go here -- it goes
in a `.env` file (see .env.example in the project root), which is never
committed to git and is read separately by src/llm_assistant.py.
"""

import os

# ----------------------------------------------------------------------
# PROJECT PATHS
# Resolved automatically -- you shouldn't need to change these.
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ----------------------------------------------------------------------
# 1. DATASET GENERATION (src/data_generator.py)
# ----------------------------------------------------------------------
RANDOM_SEED = 42                 # change this to get a different random dataset
NUM_MERCHANTS = 40
NUM_TRANSACTIONS = 3000          # increase for a bigger demo dataset, decrease for faster iteration
DATASET_START_DATE = (2026, 1, 1)   # (year, month, day)
DATASET_END_DATE = (2026, 6, 30)
PAYMENT_METHODS = ["UPI", "credit_card", "debit_card", "netbanking", "wallet"]
CURRENCY = "INR"

# Roughly what fraction of every payment goes to gateway fees / GST.
# Used to simulate realistic fee and tax amounts.
FEE_RATE_RANGE = (0.015, 0.025)   # 1.5% - 2.5% typical gateway processing fee
GST_ON_FEE_RATE = 0.18            # 18% GST on the fee amount, standard in India

# What fraction of all transactions fall into each scenario bucket.
# Must sum to 1.0 -- data_generator.py asserts this on startup.
SCENARIO_WEIGHTS = {
    "exact_match": 0.544,
    "fee_only": 0.12,
    "tax_only": 0.06,
    "fee_and_tax": 0.08,
    "legitimate_refund": 0.05,
    "normal_delay": 0.03,
    "missing_settlement": 0.02,
    "duplicate_settlement": 0.015,
    "incorrect_amount": 0.015,
    "duplicate_refund": 0.01,
    "missing_refund_record": 0.01,
    "incorrect_fee": 0.01,
    "unusual_amount_modification": 0.008,
    "repeated_manual_adjustments": 0.007,
    "suspicious_operator_activity": 0.006,
    "unusual_hours_activity": 0.006,
    "high_adjustment_frequency": 0.005,
    "suspicious_device_ip_correlation": 0.004,
}

# Simulated employees, devices, and IPs used when generating audit trail data.
NUM_OPERATORS = 10
SUSPICIOUS_OPERATORS = ["OP009", "OP010"]  # used for the injected security scenarios
NUM_DEVICES = 15
SUSPICIOUS_DEVICE = "DEV099"
SUSPICIOUS_IP = "203.0.113.77"
BULK_SUSPICIOUS_TOUCHES = 45      # how many extra transactions OP010 "touches" to simulate a compromised account


# ----------------------------------------------------------------------
# 2. RECONCILIATION (src/reconciliation.py)
# ----------------------------------------------------------------------
# Amounts within this tolerance are treated as a full match. Protects
# against floating-point rounding noise -- not a business rule.
AMOUNT_TOLERANCE = 0.01


# ----------------------------------------------------------------------
# 3. EXCEPTION CLASSIFICATION (src/exceptions.py)
# ----------------------------------------------------------------------
FULL_MATCH_TOLERANCE = 0.01

# If recorded adjustments explain at least this fraction of the total gap,
# classify as PARTIALLY_EXPLAINED instead of fully UNRESOLVED.
PARTIAL_EXPLANATION_THRESHOLD = 0.5

RECONCILED_STATUSES = {"MATCHED", "RECONCILED_WITH_ADJUSTMENT"}
EXCEPTION_STATUSES = {"DUPLICATE_SETTLEMENT", "MISSING_SETTLEMENT", "PARTIALLY_EXPLAINED", "UNRESOLVED"}


# ----------------------------------------------------------------------
# 4. ANOMALY DETECTION (src/anomaly_detection.py)
# ----------------------------------------------------------------------
ODD_HOURS_RANGE = (0, 6)  # midnight - 6am counts as "unusual hours"

# Rule-based thresholds -- change these to make the rule checks stricter/looser.
HIGH_FREQUENCY_ADJUSTMENT_COUNT = 4       # >= this many adjustments on one txn is unusual
LARGE_SINGLE_ADJUSTMENT_FRACTION = 0.3    # a single adjustment > this fraction of payment amount is unusual
SHORT_WINDOW_MINUTES = 20                 # multiple adjustments within this window = "rapid burst"
OPERATOR_ZSCORE_FLAG_THRESHOLD = 3.5      # operator activity z-score above which a rule flag fires

# Isolation Forest settings.
ISOLATION_FOREST_CONTAMINATION = 0.1      # expected fraction of outliers among transactions-with-activity
ISOLATION_FOREST_RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 5. SECURITY / ENTITY CORRELATION (src/security_analysis.py)
# ----------------------------------------------------------------------
# MAD-based robust z-score threshold for flagging an operator as a
# high-volume outlier. 3.5 is a common convention for robust z-scores
# (higher than a standard z-score threshold like 2, because MAD-based
# scores run larger -- see README section 10 for why we use MAD at all).
HIGH_VOLUME_ZSCORE = 3.5

# A device or IP used by this many or more DISTINCT operators is flagged
# as shared/suspicious (normally one device/IP maps to ~one operator).
SHARED_DEVICE_OPERATOR_THRESHOLD = 2


# ----------------------------------------------------------------------
# 6. RISK SCORING (src/risk_scoring.py)
# ----------------------------------------------------------------------
# Maximum points each component can contribute. Must be four numbers that
# make sense to you as relative weights -- they don't have to sum to
# exactly 100, but the scorer clips the final total at 100 either way.
RISK_WEIGHT_FINANCIAL = 30
RISK_WEIGHT_BEHAVIORAL = 30
RISK_WEIGHT_ML = 20
RISK_WEIGHT_ENTITY = 20

# Score cutoffs for LOW / MEDIUM / HIGH.
RISK_LOW_MEDIUM_CUTOFF = 30
RISK_MEDIUM_HIGH_CUTOFF = 60


# ----------------------------------------------------------------------
# 7. LLM INVESTIGATION ASSISTANT (src/llm_assistant.py) -- OPTIONAL
# ----------------------------------------------------------------------
# Which provider to use. Change this ONE line to switch providers -- no
# other code needs to change. Options: "ollama", "groq", "gemini", "anthropic".
#
#   ollama    -- $0 forever, no API key, runs on YOUR machine (via `ollama
#                serve`). Best for local demos. If you deploy this app
#                online, Ollama needs to run on a server you control --
#                most free hosting platforms can't run it in the background
#                for you.
#   groq      -- free tier, hosted for you, no server to manage. Good
#                default if you want "free" AND "works once deployed online".
#   gemini    -- free tier (Google), hosted for you.
#   anthropic -- Claude models. Small free trial credit, then paid.
LLM_PROVIDER = "groq"

LLM_MAX_TOKENS = 400

# --- Ollama settings (used when LLM_PROVIDER = "ollama") ---
# Install from https://ollama.com, then run `ollama serve` and
# `ollama pull llama3.2` (or any model you prefer) before using this.
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"

# --- Groq settings (used when LLM_PROVIDER = "groq") ---
# Free API key at https://console.groq.com -- set GROQ_API_KEY in .env
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-20b"

# --- Gemini settings (used when LLM_PROVIDER = "gemini") ---
# Free API key at https://aistudio.google.com/apikey -- set GEMINI_API_KEY in .env
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.0-flash"

# --- Anthropic settings (used when LLM_PROVIDER = "anthropic") ---
# API key at https://console.anthropic.com -- set ANTHROPIC_API_KEY in .env
# Cheaper/faster option: "claude-haiku-4-5-20251001". Higher quality: "claude-opus-5".
ANTHROPIC_MODEL = "claude-sonnet-5"

# The actual API keys are NEVER stored here. They're read from environment
# variables (see .env.example) so they never end up hardcoded in source
# code or committed to git.

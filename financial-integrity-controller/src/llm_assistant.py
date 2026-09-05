"""
llm_assistant.py
------------------
Optional GenAI investigation assistant.

CRITICAL DESIGN RULE (do not violate this when modifying the code):
The LLM NEVER computes reconciliation facts, risk scores, or amounts. It only
receives a JSON blob of ALREADY-COMPUTED facts (from risk_scores.csv) and is
instructed to explain them in plain English. This prevents hallucinated
numbers -- if the LLM doesn't have a fact in its prompt, it cannot invent one,
because the system prompt explicitly forbids it from stating any number that
isn't in the provided evidence.

PROVIDER-AGNOSTIC BY DESIGN:
This module works with any of four providers, chosen via LLM_PROVIDER in
config.py. Because the design constrains the model to explain pre-computed
facts rather than reason freely, ANY reasonably capable instruction-following
model works here -- there's no lock-in to a specific vendor.

  "ollama"    -- free, runs locally on your own machine, no API key needed.
                 Best for local development/demos. To use this when your app
                 is deployed online, Ollama needs to be running on a server
                 you control (a free hosting platform typically can't run it
                 for you in the background).
  "groq"      -- free hosted API with a generous free tier, fast inference,
                 open models (Llama, Gemma). Good fit if you want "free" AND
                 "hosted online" without managing your own server.
  "gemini"    -- free hosted API (Google), generous daily quota.
  "anthropic" -- Claude models. Paid after a small free trial credit.

Three of these (ollama, groq, gemini) speak an OpenAI-compatible API, so they
share one code path below and differ only in base_url/model/key. Anthropic
uses its own SDK and message format, so it has a separate code path.

Requires the relevant API key (or a running Ollama server) depending on
LLM_PROVIDER in config.py. If misconfigured, the dashboard catches the
resulting error and shows a friendly message -- the rest of the app works
fully without any of this.
"""

import os
import json

import pandas as pd

import config as cfg

# Automatically load a .env file if one exists (see .env.example) so you
# can drop your API key in a file instead of exporting it in your shell
# every time. Optional -- python-dotenv is imported lazily, and if it's
# not installed, we just skip this and fall back to whatever is already
# in the environment (e.g. a shell export or a platform's env var settings).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


SYSTEM_PROMPT = """You are a financial investigation assistant for a payments \
reconciliation and risk-scoring system.

You will be given a JSON object containing ALREADY-COMPUTED facts about one \
transaction: its reconciliation status, amounts, explanation, risk score, \
risk level, and the specific reasons behind that score.

Your job is ONLY to explain these facts clearly in plain English, in 2-4 \
sentences.

STRICT RULES:
- Do not invent, estimate, or guess any number, date, name, or fact that is \
not explicitly present in the provided JSON.
- Do not claim a transaction is confirmed fraud. Use the same cautious \
language already present in the data (e.g. "anomaly", "suspicious activity", \
"investigation required") rather than asserting fraud.
- If the user asks something the provided facts cannot answer, say so \
plainly instead of guessing.
- Be concise and factual, like a financial analyst writing a case note.
"""


def build_evidence(transaction_id, risk_df: pd.DataFrame):
    """Extracts exactly the pre-computed facts for one transaction into a
    plain dict -- this dict, and nothing else, is what the LLM is allowed
    to reason over."""
    row = risk_df[risk_df["transaction_id"] == transaction_id]
    if row.empty:
        raise ValueError(f"Transaction {transaction_id} not found.")
    row = row.iloc[0]

    fields = [
        "transaction_id", "amount", "expected_settlement", "actual_settlement",
        "difference", "recon_status", "explanation", "risk_score", "risk_level",
        "financial_points", "behavioral_points", "ml_points", "entity_points",
        "recommended_action", "reasons",
    ]
    evidence = {f: (None if pd.isna(row[f]) else row[f]) for f in fields if f in row}
    return evidence


def _build_user_message(transaction_id, evidence, question):
    return (
        f"Evidence for transaction {transaction_id}:\n"
        f"{json.dumps(evidence, indent=2, default=str)}\n\n"
        f"Question: {question}"
    )


def _answer_via_openai_compatible(base_url, api_key, model, system_prompt, user_message):
    """
    Shared code path for any provider that speaks the OpenAI chat-completions
    format: Ollama (local, free, no key), Groq (hosted, free tier), and
    Gemini (hosted, free tier via Google's OpenAI-compatible endpoint).
    """
    from openai import OpenAI  # imported lazily so the rest of the app works without this dependency

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        max_tokens=cfg.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def _answer_via_anthropic(api_key, model, system_prompt, user_message):
    import anthropic  # imported lazily so the rest of the app works without this dependency

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=cfg.LLM_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in message.content if hasattr(block, "text"))


def answer_question(transaction_id, question, risk_df: pd.DataFrame):
    """
    Calls whichever LLM provider is configured in config.LLM_PROVIDER with
    the evidence dict + user question. Raises a clear, specific error if
    the provider isn't reachable or configured -- the caller (dashboard) is
    responsible for showing a friendly message.
    """
    provider = cfg.LLM_PROVIDER.lower()
    evidence = build_evidence(transaction_id, risk_df)
    user_message = _build_user_message(transaction_id, evidence, question)

    if provider == "ollama":
        # No API key needed -- Ollama runs locally and doesn't check one.
        # The OpenAI client library requires a non-empty string anyway, so
        # we pass a harmless placeholder.
        try:
            return _answer_via_openai_compatible(
                base_url=cfg.OLLAMA_BASE_URL, api_key="ollama",
                model=cfg.OLLAMA_MODEL, system_prompt=SYSTEM_PROMPT, user_message=user_message,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {cfg.OLLAMA_BASE_URL}. Is `ollama serve` "
                f"running, and have you pulled the model with `ollama pull {cfg.OLLAMA_MODEL}`? "
                f"Original error: {e}"
            )

    elif provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set (see .env.example).")
        return _answer_via_openai_compatible(
            base_url=cfg.GROQ_BASE_URL, api_key=api_key,
            model=cfg.GROQ_MODEL, system_prompt=SYSTEM_PROMPT, user_message=user_message,
        )

    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set (see .env.example).")
        return _answer_via_openai_compatible(
            base_url=cfg.GEMINI_BASE_URL, api_key=api_key,
            model=cfg.GEMINI_MODEL, system_prompt=SYSTEM_PROMPT, user_message=user_message,
        )

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set (see .env.example).")
        return _answer_via_anthropic(
            api_key=api_key, model=cfg.ANTHROPIC_MODEL,
            system_prompt=SYSTEM_PROMPT, user_message=user_message,
        )

    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}' in config.py. "
            f"Expected one of: ollama, groq, gemini, anthropic."
        )


if __name__ == "__main__":
    # Small manual test -- requires data/processed/risk_scores.csv and
    # whichever provider is configured in config.py to be reachable/keyed.
    path = os.path.join(cfg.PROCESSED_DIR, "risk_scores.csv")
    if not os.path.exists(path):
        print("Run the pipeline first (see README) to generate risk_scores.csv")
    else:
        df = pd.read_csv(path)
        exceptions = df[df["recon_status"] != "MATCHED"]
        if len(exceptions) == 0:
            print("No exceptions found to test with.")
        else:
            txn = exceptions.iloc[0]["transaction_id"]
            print(f"Testing with transaction: {txn} (provider: {cfg.LLM_PROVIDER})")
            print(answer_question(txn, "Why is this transaction unresolved or flagged?", df))

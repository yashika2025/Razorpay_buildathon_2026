"""
dashboard/app.py
------------------
Streamlit dashboard for TallyQ -- an AI Financial Integrity & Risk Controller.

This is a thin presentation layer only -- it does NOT recompute anything on
its own. It reads the CSV outputs already produced by the src/ pipeline
(data_generator -> data_validator -> reconciliation -> exceptions ->
anomaly_detection -> security_analysis -> risk_scoring) and displays them.

Run with:  streamlit run dashboard/app.py
(run the pipeline scripts in src/ first if data/processed/ is empty --
there's a "Run full pipeline now" button in the sidebar for convenience,
or use the "Upload Data" section to bring your own CSVs.)

NOTE ON THIS FILE'S STYLING: all custom CSS/HTML below is purely visual --
none of it touches data loading, computation, or business logic. Every
number, filter, and table shown here still comes straight from the same
CSVs and the same columns as before. The new Upload Data section does not
change any pipeline script in src/ either -- it just writes files into the
same data/raw/ folder those scripts already read from, then calls them.
"""

import os
import sys
import base64
import subprocess
import shutil

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
sys.path.insert(0, SRC_DIR)


def asset_path(filename):
    return os.path.join(ASSETS_DIR, filename)


@st.cache_data
def b64_image(filename):
    """Base64-encodes a local asset image so it can be embedded directly
    inside custom HTML/CSS blocks (banners, hero sections, etc.)."""
    path = asset_path(filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

import config as cfg  # noqa: E402 -- must come after sys.path.insert above
RAW_DIR = cfg.RAW_DIR
PROCESSED_DIR = cfg.PROCESSED_DIR

st.set_page_config(
    page_title="TallyQ -- AI Financial Integrity & Risk Controller",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# STYLING ONLY -- no data, no logic. Just CSS injected into the page.
# ----------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
        :root {
            --fic-navy: #0f172a;
            --fic-navy-light: #1e293b;
            --fic-accent: #6366f1;
            --fic-accent-light: #818cf8;
            --fic-green: #16a34a;
            --fic-amber: #d97706;
            --fic-red: #dc2626;
            --fic-text-muted: #64748b;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--fic-navy) 0%, var(--fic-navy-light) 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            background: var(--fic-accent);
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: var(--fic-accent-light);
        }

        h1 {
            font-weight: 800 !important;
            color: var(--fic-navy) !important;
            letter-spacing: -0.02em;
        }
        h2, h3 {
            font-weight: 700 !important;
            color: var(--fic-navy) !important;
        }

        .fic-card {
            background: white;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04);
            border: 1px solid #e2e8f0;
            margin-bottom: 8px;
        }
        .fic-card-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--fic-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }
        .fic-card-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--fic-navy);
        }
        .fic-card-accent { border-left: 4px solid var(--fic-accent); }
        .fic-card-green { border-left: 4px solid var(--fic-green); }
        .fic-card-amber { border-left: 4px solid var(--fic-amber); }
        .fic-card-red { border-left: 4px solid var(--fic-red); }

        .fic-badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .fic-badge-high { background: #fee2e2; color: var(--fic-red); }
        .fic-badge-medium { background: #fef3c7; color: var(--fic-amber); }
        .fic-badge-low { background: #dcfce7; color: var(--fic-green); }

        .fic-reason {
            background: #fafaf9;
            border: 1px solid #e7e5e4;
            border-left: 3px solid var(--fic-accent);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-size: 0.92rem;
            color: #292524;
        }

        .fic-amount-box {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px;
            text-align: center;
        }
        .fic-amount-label {
            font-size: 0.75rem;
            color: var(--fic-text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }
        .fic-amount-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--fic-navy);
        }

        .fic-upload-box {
            background: white;
            border: 1.5px dashed #c7d2fe;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 10px;
        }
        .fic-upload-title {
            font-weight: 700;
            color: var(--fic-navy);
            font-size: 0.95rem;
        }
        .fic-upload-required {
            color: var(--fic-red);
            font-size: 0.75rem;
            font-weight: 700;
        }
        .fic-upload-optional {
            color: var(--fic-text-muted);
            font-size: 0.75rem;
        }

        hr { margin: 1.2rem 0 !important; }

        .fic-sidebar-title {
            font-size: 1.3rem;
            font-weight: 800;
            line-height: 1.3;
            margin-bottom: 2px;
        }
        .fic-sidebar-subtitle {
            font-size: 0.85rem;
            font-weight: 500;
            color: #cbd5e1 !important;
            margin-bottom: 2px;
        }
        .fic-sidebar-caption {
            font-size: 0.78rem;
            color: #94a3b8 !important;
            margin-bottom: 18px;
        }

        /* Mascot robot in sidebar */
        .fic-mascot-wrap {
            display: flex;
            justify-content: center;
            margin-bottom: 6px;
        }
        section[data-testid="stSidebar"] .fic-mascot-wrap img {
            border-radius: 50%;
            border: 3px solid var(--fic-accent-light);
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.18);
        }

        /* ---------------- Pipeline flow stepper ---------------- */
        .fic-flow-container {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 2px;
            margin: 10px 0 18px 0;
            padding: 16px 10px;
            background: white;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
        }
        .fic-flow-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-width: 92px;
            padding: 10px 6px;
            border-radius: 12px;
            background: #f8fafc;
            border: 1.5px solid #e2e8f0;
            transition: all .25s ease;
        }
        .fic-flow-icon { font-size: 1.5rem; line-height: 1; }
        .fic-flow-label {
            font-size: 0.66rem;
            font-weight: 700;
            color: var(--fic-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.02em;
            margin-top: 4px;
            text-align: center;
        }
        .fic-flow-done { background: #dcfce7; border-color: var(--fic-green); }
        .fic-flow-done .fic-flow-label { color: var(--fic-green); }
        .fic-flow-active {
            background: #e0e7ff;
            border-color: var(--fic-accent);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
            animation: fic-pulse 1.4s ease-in-out infinite;
        }
        .fic-flow-active .fic-flow-label { color: var(--fic-accent); }
        .fic-flow-skip { opacity: 0.4; }
        .fic-flow-arrow { color: #cbd5e1; font-size: 1.15rem; padding: 0 2px; }
        .fic-flow-arrow-done { color: var(--fic-green); }
        @keyframes fic-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.35); }
            70%  { box-shadow: 0 0 0 9px rgba(99, 102, 241, 0); }
            100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        }

        /* ---------------- Get Started / landing screen ---------------- */
        .fic-hero {
            position: relative;
            display: flex;
            align-items: center;
            gap: 22px;
            background: linear-gradient(120deg, #0f172a 0%, #1e293b 55%, #312e81 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
            overflow: hidden;
        }
        .fic-hero img.fic-hero-mascot {
            border-radius: 50%;
            border: 3px solid #818cf8;
            box-shadow: 0 0 0 6px rgba(129, 140, 248, 0.18);
            flex-shrink: 0;
        }
        .fic-hero h1 { color: white !important; margin: 0 0 6px 0 !important; }
        .fic-hero p { color: #cbd5e1; margin: 0; font-size: 0.95rem; }

        .fic-choice-card {
            background: white;
            border-radius: 0 0 16px 16px;
            border: 1.5px solid #e2e8f0;
            border-top: none;
            padding: 18px 20px 20px 20px;
            text-align: center;
        }
        .fic-choice-card h3 { margin: 4px 0 8px 0 !important; }
        .fic-choice-desc {
            color: var(--fic-text-muted);
            font-size: 0.88rem;
            margin-bottom: 4px;
            min-height: 56px;
        }
        .fic-choice-img {
            width: 100%;
            height: 140px;
            object-fit: cover;
            border-radius: 16px 16px 0 0;
            display: block;
        }

        /* ---------------- Investigation hero banner ---------------- */
        .fic-invest-hero {
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 20px;
            height: 150px;
        }
        .fic-invest-hero img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: brightness(0.4) saturate(1.1);
        }
        .fic-invest-hero-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 0 30px;
        }
        .fic-invest-hero-overlay h1 { color: white !important; margin: 0 0 4px 0 !important; }
        .fic-invest-hero-overlay p { color: #e2e8f0 !important; margin: 0; font-size: 0.92rem; }

        .fic-risk-gauge-thumb {
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        .fic-assistant-avatar {
            border-radius: 50%;
            border: 2px solid var(--fic-accent-light);
        }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label, value, accent="accent"):
    """Renders one custom metric card. Purely visual -- `value` is passed
    in exactly as computed elsewhere, never altered here."""
    st.markdown(f"""
    <div class="fic-card fic-card-{accent}">
        <div class="fic-card-label">{label}</div>
        <div class="fic-card-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def risk_badge_html(level):
    """Returns a small colored pill for a risk level string. Purely visual."""
    level = str(level).upper()
    cls = {"HIGH": "fic-badge-high", "MEDIUM": "fic-badge-medium", "LOW": "fic-badge-low"}.get(level, "fic-badge-low")
    icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(level, "⚪")
    return f'<span class="fic-badge {cls}">{icon} {level}</span>'


inject_custom_css()


# ----------------------------------------------------------------------
# DATA LOADING (unchanged from before -- purely functional, no styling)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads every processed CSV the dashboard needs. Returns None for any
    file that doesn't exist yet, so the UI can prompt the user to run the
    pipeline instead of crashing."""
    def try_read(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    return {
        "risk_scores": try_read(os.path.join(PROCESSED_DIR, "risk_scores.csv")),
        "reconciliation": try_read(os.path.join(PROCESSED_DIR, "reconciliation.csv")),
        "anomaly_features": try_read(os.path.join(PROCESSED_DIR, "anomaly_features.csv")),
        "refunds": try_read(os.path.join(RAW_DIR, "refunds.csv")),
        "fees": try_read(os.path.join(RAW_DIR, "fees.csv")),
        "taxes": try_read(os.path.join(RAW_DIR, "taxes.csv")),
        "adjustments": try_read(os.path.join(RAW_DIR, "adjustments.csv")),
        "audit_logs": try_read(os.path.join(RAW_DIR, "audit_logs.csv")),
        "payments": try_read(os.path.join(RAW_DIR, "payments.csv")),
    }


def run_full_pipeline():
    """Runs every stage of the pipeline in order, in-process, so the
    dashboard can bootstrap itself with one click during a demo. This
    INCLUDES data_generator.py, so it will overwrite data/raw/ with a fresh
    synthetic dataset -- use this when you want demo data, not your own."""
    scripts = [
        "data_generator.py", "data_validator.py", "reconciliation.py",
        "exceptions.py", "anomaly_detection.py", "security_analysis.py",
        "risk_scoring.py",
    ]
    return _run_scripts(scripts, skip_indices=set())


def run_pipeline_on_existing_data():
    """Runs every stage EXCEPT data_generator.py, so whatever is already
    sitting in data/raw/ (e.g. files you just uploaded) is processed as-is,
    without being overwritten by a freshly generated synthetic dataset."""
    scripts = [
        "data_validator.py", "reconciliation.py", "exceptions.py",
        "anomaly_detection.py", "security_analysis.py", "risk_scoring.py",
    ]
    return _run_scripts(scripts, skip_indices={0})


# Visual pipeline stages shown in the flow indicator. Purely cosmetic --
# mirrors the actual script order above but adds a friendly label/icon.
PIPELINE_STAGES = [
    ("Generate", "🧬"),
    ("Validate", "✅"),
    ("Reconcile", "📒"),
    ("Exceptions", "⚠️"),
    ("Anomaly Scan", "🔎"),
    ("Security Scan", "🛡️"),
    ("Risk Scoring", "🎯"),
]
SCRIPT_STAGE_INDEX = {
    "data_generator.py": 0,
    "data_validator.py": 1,
    "reconciliation.py": 2,
    "exceptions.py": 3,
    "anomaly_detection.py": 4,
    "security_analysis.py": 5,
    "risk_scoring.py": 6,
}


def render_pipeline_flow(active_index=None, done_index=-1, skip_indices=None, placeholder=None):
    """Renders a horizontal flow/stepper showing the 7 pipeline stages.

    active_index : stage currently running (pulses)
    done_index   : highest stage index already completed (shown green)
    skip_indices : stages intentionally skipped this run (e.g. data
                   generation, when processing an uploaded batch) -- shown
                   dimmed rather than red/failed.
    placeholder  : an st.empty() to render into, so this can be updated
                   in-place across pipeline steps instead of stacking.
    """
    skip_indices = skip_indices or set()
    parts = []
    n = len(PIPELINE_STAGES)
    for i, (label, icon) in enumerate(PIPELINE_STAGES):
        if i in skip_indices:
            cls = "fic-flow-step fic-flow-skip"
        elif i <= done_index:
            cls = "fic-flow-step fic-flow-done"
        elif i == active_index:
            cls = "fic-flow-step fic-flow-active"
        else:
            cls = "fic-flow-step"
        parts.append(
            f'<div class="{cls}"><div class="fic-flow-icon">{icon}</div>'
            f'<div class="fic-flow-label">{label}</div></div>'
        )
        if i < n - 1:
            arrow_cls = "fic-flow-arrow fic-flow-arrow-done" if i < done_index else "fic-flow-arrow"
            parts.append(f'<div class="{arrow_cls}">➜</div>')
    html = f'<div class="fic-flow-container">{"".join(parts)}</div>'
    target = placeholder if placeholder is not None else st
    target.markdown(html, unsafe_allow_html=True)


def _run_scripts(scripts, skip_indices=None):
    """Runs the given pipeline scripts in order, updating the visual flow
    stepper as it goes. Also measures real wall-clock time and counts the
    rows actually processed (from payments.csv, once it exists) so the UI
    can report genuine throughput -- not a guess."""
    import time

    skip_indices = skip_indices or set()
    flow_slot = st.empty()
    status_slot = st.empty()
    render_pipeline_flow(active_index=None, done_index=-1, skip_indices=skip_indices, placeholder=flow_slot)

    start_time = time.perf_counter()
    completed = max(skip_indices) if skip_indices else -1
    for script in scripts:
        stage_idx = SCRIPT_STAGE_INDEX.get(script)
        status_slot.markdown(f"⏳ Running **{script}** ...")
        render_pipeline_flow(active_index=stage_idx, done_index=completed, skip_indices=skip_indices, placeholder=flow_slot)
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, script)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            status_slot.empty()
            st.error(f"{script} failed:\n{result.stderr}")
            return False
        completed = stage_idx if stage_idx is not None else completed
    elapsed = time.perf_counter() - start_time

    payments_path = os.path.join(RAW_DIR, "payments.csv")
    try:
        record_count = len(pd.read_csv(payments_path, usecols=[0]))
    except Exception:
        record_count = None

    render_pipeline_flow(active_index=None, done_index=len(PIPELINE_STAGES) - 1, skip_indices=skip_indices, placeholder=flow_slot)
    if record_count:
        rate = record_count / elapsed if elapsed > 0 else 0
        status_slot.success(
            f"✅ Pipeline complete -- processed **{record_count:,} records** in "
            f"**{elapsed:.2f}s** (~{rate:,.0f} records/sec)."
        )
        st.session_state["last_pipeline_stats"] = {
            "records": record_count, "seconds": round(elapsed, 2), "rate": round(rate, 1),
        }
    else:
        status_slot.success(f"✅ Pipeline complete in {elapsed:.2f}s.")
    return True


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
agent_b64 = b64_image("agent_robot.jpg")
if agent_b64:
    st.sidebar.markdown(
        f'<div class="fic-mascot-wrap"><img src="data:image/jpeg;base64,{agent_b64}" width="84"/></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    '<div class="fic-sidebar-title">🛡️ TallyQ</div>'
    '<div class="fic-sidebar-subtitle">AI Financial Integrity & Risk Controller</div>'
    '<div class="fic-sidebar-caption">Razorpay AI Buildathon 2026 — Finance Controller track</div>',
    unsafe_allow_html=True,
)

if st.sidebar.button("🔄  Run full pipeline (demo data)"):
    success = run_full_pipeline()
    if success:
        st.cache_data.clear()
        st.session_state.nav_target = "Overview"
        st.rerun()

if st.sidebar.button("🏁  Choose data source again"):
    st.session_state.nav_target = "Get Started"
    st.rerun()

st.sidebar.markdown("---")

data = load_data()

PAGE_ICONS = {
    "Get Started": "🏁",
    "Overview": "📊",
    "Reconciliation": "📒",
    "Security / Integrity": "🛡️",
    "Investigation": "🔍",
    "Upload Data": "📤",
}

# NOTE: nav_page is bound to the sidebar radio's `key` below, so Streamlit
# forbids writing to it directly once that widget has been instantiated in
# a run. Any "redirect to another page" action instead sets nav_target and
# calls st.rerun(); on the *next* run we consume nav_target here, before
# the radio widget is created, which Streamlit allows.
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Get Started"
if "nav_target" in st.session_state:
    st.session_state.nav_page = st.session_state.pop("nav_target")

page = st.sidebar.radio(
    "SECTION",
    list(PAGE_ICONS.keys()),
    format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
    key="nav_page",
)

if data["risk_scores"] is None and page not in ("Upload Data", "Get Started"):
    st.warning(
        "No processed data found yet. Click **'Run full pipeline (demo data)'** in the "
        "sidebar, use the **📤 Upload Data** section to bring your own CSVs, or head back "
        "to **🏁 Get Started** to choose how you'd like to begin."
    )
    st.stop()

risk_df = data["risk_scores"]

RECONCILED_STATUSES = {"MATCHED", "RECONCILED_WITH_ADJUSTMENT"}

# ----------------------------------------------------------------------
# GET STARTED -- choose demo data or upload your own, then get redirected
# straight into the next step. Purely a navigation/UX convenience; it does
# not change what the pipeline computes, only when/how it's triggered.
# ----------------------------------------------------------------------
if page == "Get Started":
    hero_b64 = b64_image("agent_robot.jpg")
    st.markdown(
        f"""
        <div class="fic-hero">
            <img class="fic-hero-mascot" src="data:image/jpeg;base64,{hero_b64}" width="80"/>
            <div>
                <h1>Welcome to TallyQ 👋</h1>
                <p>Your AI Financial Integrity & Risk Controller. Choose how you'd like to
                start -- everything below runs through the exact same reconciliation,
                anomaly-detection, and risk-scoring engine either way.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    demo_b64 = b64_image("money2.jpg")
    upload_b64 = b64_image("money1.jpg")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <img class="fic-choice-img" src="data:image/jpeg;base64,{demo_b64}"/>
            <div class="fic-choice-card">
                <h3>📊 Try Demo Data</h3>
                <div class="fic-choice-desc">Instantly generate a realistic synthetic dataset
                of payments, settlements, refunds, fees and more -- the fastest way to see
                TallyQ in action.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀  Use Demo Data", use_container_width=True, key="choose_demo"):
            success = run_full_pipeline()
            if success:
                st.cache_data.clear()
                st.session_state.nav_target = "Overview"
                st.rerun()

    with c2:
        st.markdown(
            f"""
            <img class="fic-choice-img" src="data:image/jpeg;base64,{upload_b64}"/>
            <div class="fic-choice-card">
                <h3>📤 Upload My Own Data</h3>
                <div class="fic-choice-desc">Bring your own payments, settlements and related
                CSVs. They'll be validated and processed by the same engine as the demo
                dataset.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📁  Go to Upload Data", use_container_width=True, key="choose_upload"):
            st.session_state.nav_target = "Upload Data"
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("🔗 How the pipeline works")
    render_pipeline_flow(active_index=None, done_index=-1)
    st.caption(
        "Every stage above runs in this order for whichever data source you pick -- "
        "validation, reconciliation, exception detection, anomaly scanning, security "
        "checks, then final risk scoring."
    )

# ----------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------
elif page == "Overview":
    st.title("📊 Overview")

    pipeline_stats = st.session_state.pop("last_pipeline_stats", None)
    if pipeline_stats:
        st.success(
            f"⚡ Just processed **{pipeline_stats['records']:,} records** in "
            f"**{pipeline_stats['seconds']}s** (~{pipeline_stats['rate']:,.0f} records/sec)."
        )

    total = len(risk_df)
    reconciled = risk_df["recon_status"].isin(RECONCILED_STATUSES).sum()
    exceptions = total - reconciled
    match_rate = round(100 * reconciled / total, 2) if total else 0
    unresolved = (risk_df["recon_status"] == "UNRESOLVED").sum()
    high_risk = (risk_df["risk_level"] == "HIGH").sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total Transactions", f"{total:,}", accent="accent")
    with c2:
        metric_card("Reconciled", f"{reconciled:,}", accent="green")
    with c3:
        metric_card("Match Rate", f"{match_rate}%", accent="green")

    c4, c5, c6 = st.columns(3)
    with c4:
        metric_card("Exceptions", f"{exceptions:,}", accent="amber")
    with c5:
        metric_card("Unresolved Exceptions", f"{unresolved:,}", accent="amber")
    with c6:
        metric_card("High-Risk Exceptions", f"{high_risk:,}", accent="red")

    # ---- Cash Position -------------------------------------------------
    # How much rupee value is currently NOT accounted for by the books,
    # right now -- derived purely from already-computed columns
    # (recon_status, difference, expected_settlement). A MISSING_SETTLEMENT
    # transaction has its entire expected settlement unaccounted for (no
    # money has been confirmed to move yet); every other unresolved/
    # partially-explained exception contributes the absolute gap that no
    # recorded fee/tax/refund/adjustment explains.
    def _unaccounted_amount(row):
        if row["recon_status"] == "MISSING_SETTLEMENT":
            return row["expected_settlement"] if pd.notna(row["expected_settlement"]) else 0
        if pd.notna(row["difference"]):
            return abs(row["difference"])
        return 0

    exception_mask = ~risk_df["recon_status"].isin(RECONCILED_STATUSES)
    cash_at_risk = risk_df.loc[exception_mask].apply(_unaccounted_amount, axis=1).sum() if exception_mask.any() else 0.0
    total_volume = risk_df["amount"].sum()
    cash_at_risk_pct = round(100 * cash_at_risk / total_volume, 3) if total_volume else 0.0

    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("💵 Cash Position")
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        metric_card("Total Payment Volume", f"₹{total_volume:,.2f}", accent="accent")
    with cp2:
        metric_card("Cash Currently Unaccounted For", f"₹{cash_at_risk:,.2f}", accent="red")
    with cp3:
        metric_card("Unaccounted as % of Volume", f"{cash_at_risk_pct}%", accent="amber")
    st.caption(
        "\"Unaccounted for\" = the rupee amount across every exception (UNRESOLVED, "
        "PARTIALLY_EXPLAINED, MISSING_SETTLEMENT, DUPLICATE_SETTLEMENT) that no recorded "
        "fee, tax, refund, or adjustment currently explains -- the actual money finance-ops "
        "still needs to chase down, not just a transaction count."
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Reconciliation status breakdown")
    status_counts = risk_df["recon_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    st.bar_chart(status_counts.set_index("Status"), color="#6366f1")

    st.subheader("Risk level breakdown (exceptions only)")
    exc_df = risk_df[~risk_df["recon_status"].isin(RECONCILED_STATUSES)]
    if len(exc_df) > 0:
        risk_counts = exc_df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        st.bar_chart(risk_counts, color="#d97706")
    else:
        st.info("No exceptions found.")

    st.caption(
        "Note: anomaly detection here is unsupervised (Isolation Forest + rule-based "
        "signals). These results identify statistically unusual patterns for human "
        "review -- they are not a claim that any transaction is confirmed fraud."
    )


# ----------------------------------------------------------------------
# RECONCILIATION
# ----------------------------------------------------------------------
elif page == "Reconciliation":
    st.title("📒 Reconciliation")

    status_filter = st.multiselect(
        "Filter by status",
        options=sorted(risk_df["recon_status"].unique()),
        default=[],
    )
    view_df = risk_df.copy()
    if status_filter:
        view_df = view_df[view_df["recon_status"].isin(status_filter)]

    display_cols = [
        "transaction_id", "amount", "expected_settlement", "actual_settlement",
        "difference", "explanation", "recon_status",
    ]
    display_cols = [c for c in display_cols if c in view_df.columns]
    st.dataframe(
        view_df[display_cols].rename(columns={
            "amount": "Payment Amount", "expected_settlement": "Expected",
            "actual_settlement": "Actual", "difference": "Difference",
            "explanation": "Explanation", "recon_status": "Status",
        }),
        use_container_width=True, hide_index=True,
    )


# ----------------------------------------------------------------------
# SECURITY / INTEGRITY
# ----------------------------------------------------------------------
elif page == "Security / Integrity":
    gauge_col, title_col = st.columns([1, 6])
    with gauge_col:
        st.image(asset_path("risk_gauge.png"), width=110)
    with title_col:
        st.title("🛡️ Security / Integrity")
        st.caption(
            "Every exception and non-LOW-risk transaction, ranked by risk score -- "
            "the queue a fraud/finance-ops reviewer would work through first."
        )

    exc_df = risk_df[~risk_df["recon_status"].isin(RECONCILED_STATUSES) | (risk_df["risk_level"] != "LOW")]
    exc_df = exc_df.sort_values("risk_score", ascending=False)

    level_filter = st.multiselect(
        "Filter by risk level", options=["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"]
    )
    view_df = exc_df[exc_df["risk_level"].isin(level_filter)] if level_filter else exc_df

    anomaly = data["anomaly_features"]
    op_map = {}
    if anomaly is not None:
        op_map = anomaly.set_index("transaction_id")["operator_id"].to_dict()

    display_rows = []
    for _, row in view_df.iterrows():
        level = row["risk_level"]
        icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(level, "⚪")
        display_rows.append({
            "Transaction ID": row["transaction_id"],
            "Risk Score": row["risk_score"],
            "Risk Level": f"{icon} {level}",
            "Operator": op_map.get(row["transaction_id"], "-"),
            "Anomaly Reason": row["reasons"],
            "Recommended Action": row["recommended_action"],
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# INVESTIGATION
# ----------------------------------------------------------------------
elif page == "Investigation":
    invest_hero_b64 = b64_image("money1.jpg")
    st.markdown(
        f"""
        <div class="fic-invest-hero">
            <img src="data:image/jpeg;base64,{invest_hero_b64}"/>
            <div class="fic-invest-hero-overlay">
                <h1>🔍 Investigation</h1>
                <p>Drill into any unresolved transaction -- amounts, root cause, related
                records, and risk signals, all in one place.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    exception_ids = risk_df.loc[
        ~risk_df["recon_status"].isin(RECONCILED_STATUSES), "transaction_id"
    ].tolist()

    if not exception_ids:
        st.info("No exceptions to investigate -- everything reconciled cleanly.")
        st.stop()

    selected_txn = st.selectbox("Select a transaction to investigate", exception_ids)
    row = risk_df[risk_df["transaction_id"] == selected_txn].iloc[0]

    st.markdown(
        f"### Transaction `{selected_txn}` &nbsp; {risk_badge_html(row['risk_level'])}",
        unsafe_allow_html=True,
    )
    st.caption(row["recon_status"].replace("_", " ").title())

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**💰 Amounts**")
    a1, a2, a3, a4 = st.columns(4)
    actual_str = f"₹{row['actual_settlement']:,.2f}" if pd.notna(row["actual_settlement"]) else "N/A"
    diff_str = f"₹{row['difference']:,.2f}" if pd.notna(row["difference"]) else "N/A"
    for col, label, value in [
        (a1, "Payment", f"₹{row['amount']:,.2f}"),
        (a2, "Expected Settlement", f"₹{row['expected_settlement']:,.2f}"),
        (a3, "Actual Settlement", actual_str),
        (a4, "Difference", diff_str),
    ]:
        with col:
            st.markdown(f"""
            <div class="fic-amount-box">
                <div class="fic-amount-label">{label}</div>
                <div class="fic-amount-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Explanation**")
        st.info(row["explanation"])

    with col2:
        gauge_col, text_col = st.columns([1, 3])
        with gauge_col:
            st.image(asset_path("risk_gauge.png"), width=72)
        with text_col:
            st.markdown(f"**🎯 Risk Assessment — {row['risk_score']} / 100**")
            st.write(f"Recommended action: **{row['recommended_action']}**")

    st.markdown("**⚠️ Reasons**")
    for reason in str(row["reasons"]).split(" | "):
        st.markdown(f'<div class="fic-reason">⚠️ {reason}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**🔗 Related records**")

    tabs = st.tabs(["💸 Refunds", "💳 Fees", "🧾 Taxes", "✏️ Adjustments", "📜 Audit Log"])
    related = {
        "Refunds": data["refunds"], "Fees": data["fees"], "Taxes": data["taxes"],
        "Adjustments": data["adjustments"],
    }
    for tab, (name, df) in zip(tabs[:4], related.items()):
        with tab:
            if df is not None and "transaction_id" in df.columns:
                sub = df[df["transaction_id"] == selected_txn]
                if len(sub):
                    st.dataframe(sub, use_container_width=True, hide_index=True)
                else:
                    st.caption("None found.")
            else:
                st.caption("No data available.")

    with tabs[4]:
        audit = data["audit_logs"]
        if audit is not None and "entity_id" in audit.columns:
            sub = audit[audit["entity_id"] == selected_txn]
            if len(sub):
                st.dataframe(sub, use_container_width=True, hide_index=True)
            else:
                st.caption("None found.")
        else:
            st.caption("No data available.")

    st.divider()
    avatar_col, header_col = st.columns([1, 10])
    with avatar_col:
        st.image(asset_path("agent_robot.jpg"), width=48)
    with header_col:
        st.markdown(f"**🤖 Ask the investigation assistant** &nbsp; `provider: {cfg.LLM_PROVIDER}`")
    question = st.text_input("e.g. 'Why is this transaction unresolved?' or 'Why was this classified as high risk?'")
    if question:
        try:
            from llm_assistant import answer_question
            with st.spinner("Thinking..."):
                answer = answer_question(selected_txn, question, risk_df)
            st.success(answer)
        except Exception as e:
            st.error(
                f"Investigation assistant unavailable ({e}). "
                f"Check LLM_PROVIDER in src/config.py and the matching setup in .env / Ollama."
            )


# ----------------------------------------------------------------------
# UPLOAD DATA
# ----------------------------------------------------------------------
elif page == "Upload Data":
    st.title("📤 Upload Data")
    st.write(
        "Bring your own transaction records instead of using the generated demo "
        "dataset. Upload CSVs matching the schema below, then run the pipeline on "
        "them -- the exact same reconciliation, anomaly detection, and risk "
        "scoring engine used everywhere else in TallyQ will process your files."
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Each entry: (raw filename, required columns shown as a hint, is it required)
    UPLOAD_SPECS = [
        ("payments.csv", "transaction_id, merchant_id, amount, currency, timestamp, status", True),
        ("settlements.csv", "settlement_id, transaction_id, settlement_amount, settlement_timestamp, settlement_status", True),
        ("merchants.csv", "merchant_id, name", True),
        ("fees.csv", "fee_id, transaction_id, fee_amount", False),
        ("taxes.csv", "tax_id, transaction_id, tax_amount", False),
        ("refunds.csv", "refund_id, transaction_id, refund_amount, refund_timestamp", False),
        ("adjustments.csv", "adjustment_id, transaction_id, adjustment_amount, operator_id, timestamp", False),
        ("audit_logs.csv", "log_id, entity_type, entity_id, operator_id, action, timestamp, device_id, ip_address", False),
    ]

    uploaded_files = {}
    left, right = st.columns(2)
    columns_cycle = [left, right]

    for i, (filename, hint_cols, required) in enumerate(UPLOAD_SPECS):
        target_col = columns_cycle[i % 2]
        with target_col:
            badge = '<span class="fic-upload-required">REQUIRED</span>' if required else '<span class="fic-upload-optional">optional</span>'
            st.markdown(f"""
            <div class="fic-upload-box">
                <div class="fic-upload-title">{filename} &nbsp; {badge}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Expected columns: {hint_cols}")
            uploaded_files[filename] = st.file_uploader(
                f"Upload {filename}", type=["csv"], key=f"upload_{filename}",
                label_visibility="collapsed",
            )

    st.markdown("<br/>", unsafe_allow_html=True)

    required_missing = [
        fname for fname, _, required in UPLOAD_SPECS
        if required and uploaded_files.get(fname) is None
    ]

    if required_missing:
        st.caption(f"Still need: {', '.join(required_missing)} before you can process this batch.")

    process_clicked = st.button(
        "⚙️  Save uploaded files and run the pipeline",
        disabled=bool(required_missing),
    )

    if process_clicked:
        os.makedirs(RAW_DIR, exist_ok=True)

        from data_validator import REQUIRED_COLUMNS

        # Any raw file NOT re-uploaded this time (e.g. an optional file like
        # taxes.csv) is replaced with an empty file containing just the
        # correct header row -- not deleted outright. This keeps the schema
        # check in data_validator.py satisfied (it checks that expected
        # columns exist, separately from whether the file has any rows) and
        # correctly represents "this batch simply has none of this record
        # type" rather than "this file is missing/broken".
        for filename, _, _ in UPLOAD_SPECS:
            dest_path = os.path.join(RAW_DIR, filename)
            uploaded = uploaded_files.get(filename)
            table_name = filename.replace(".csv", "")
            if uploaded is not None:
                with open(dest_path, "wb") as f:
                    f.write(uploaded.getbuffer())
            else:
                header_cols = REQUIRED_COLUMNS.get(table_name, [])
                pd.DataFrame(columns=header_cols).to_csv(dest_path, index=False)

        # scenario_labels.csv is only ever produced by the synthetic
        # data_generator.py for our own internal evaluation -- it has no
        # place next to real uploaded data, so clear it if present.
        stale_labels_path = os.path.join(RAW_DIR, "scenario_labels.csv")
        if os.path.exists(stale_labels_path):
            os.remove(stale_labels_path)

        success = run_pipeline_on_existing_data()

        if success:
            st.cache_data.clear()
            st.session_state.nav_target = "Overview"
            st.rerun()

    st.divider()
    st.caption(
        "Note: uploading a new batch here replaces the dataset used across every "
        "section of TallyQ (Overview, Reconciliation, Security/Integrity, "
        "Investigation) -- it runs through the exact same validation, "
        "reconciliation, anomaly detection, and risk-scoring engine as the "
        "generated demo data. Use 'Run full pipeline (demo data)' in the sidebar "
        "at any time to switch back to the synthetic dataset."
    )

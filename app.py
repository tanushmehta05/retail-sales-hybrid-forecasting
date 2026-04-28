from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

MODEL_PATH = SOURCE_DIR / "hybrid_stacking_model.joblib"
METRICS_PATH = SOURCE_DIR / "metrics.json"
META_PATH = SOURCE_DIR / "meta_coefficients.json"
COMPARISON_PATH = SOURCE_DIR / "model_comparison_metrics.csv"


PAGE_OPTIONS = [
    "Home",
    "One-Slide Summary",
    "Project Walkthrough",
    "Report Alignment",
    "Training Pipeline Flowchart",
    "Results Dashboard",
    "Hybrid Model Explanation",
    "Prediction Demo",
    "About / Viva",
]


FLOW_STEPS = [
    {
        "icon": "📦",
        "title": "Raw Kaggle Dataset",
        "subtitle": "Walmart Recruiting - Store Sales Forecasting",
        "what": "The project starts with the Kaggle Walmart sales dataset containing weekly sales, store metadata, and external features.",
        "why": "Retail forecasting depends on historical sales plus contextual variables such as holidays, store type, fuel price, CPI, and unemployment.",
        "report": "This matches the report's proposed retail sales forecasting dataset foundation.",
    },
    {
        "icon": "🔗",
        "title": "Data Merging",
        "subtitle": "train.csv + stores.csv + features.csv",
        "what": "The sales table is merged with store information and external feature data using Store and Date keys.",
        "why": "One modeling table is easier to clean, explore, engineer, split, and pass into machine learning models.",
        "report": "The report described forecasting with combined retail and external features, so this prepares the same modeling view.",
    },
    {
        "icon": "📊",
        "title": "EDA",
        "subtitle": "sales trend, missing values, store/dept analysis",
        "what": "The notebook inspects date coverage, unique stores, departments, missing values, and sales patterns over time.",
        "why": "EDA shows seasonality, holiday spikes, data quality issues, and model-relevant patterns before training.",
        "report": "This supports the report's methodology by validating that the dataset has forecasting structure.",
    },
    {
        "icon": "🧩",
        "title": "Feature Engineering",
        "subtitle": "calendar, holiday, Indian festival, monsoon",
        "what": "Calendar features, holiday flags, Indian festival indicators, and monsoon flags are created from the date column.",
        "why": "Retail demand changes with seasonal, cultural, and weather-like context. These features expose those signals to the models.",
        "report": "The report focused on improving forecast accuracy with meaningful predictors. These features extend that idea.",
    },
    {
        "icon": "⏱️",
        "title": "Lag and Rolling Features",
        "subtitle": "Sales_Lag_1, Sales_Lag_4, Sales_Lag_12, Sales_RollMean_4, Sales_RollStd_12",
        "what": "Past weekly sales values and rolling statistics are created per Store and Dept using only earlier observations.",
        "why": "Lag features tell the model recent demand level, while rolling features summarize short-term trend and volatility.",
        "report": "This strengthens the report's forecasting approach by adding leakage-safe time-series memory.",
    },
    {
        "icon": "🗓️",
        "title": "Time-Based Split",
        "subtitle": "train before 2012, validate on 2012",
        "what": "Training uses records before 2012, and validation uses 2012 records.",
        "why": "Forecasting should learn from the past and predict the future, so random splitting is avoided.",
        "report": "This makes the evaluation more realistic for the report's retail forecasting goal.",
    },
    {
        "icon": "🌳",
        "title": "Base Model Training",
        "subtitle": "CatBoost + LightGBM",
        "what": "CatBoost and LightGBM are trained as gradient boosting base models.",
        "why": "Both are strong tabular forecasting models. CatBoost handles categorical features well, while LightGBM is fast and accurate.",
        "report": "The report proposed gradient boosting models including LightGBM and CatBoost, so this section directly aligns.",
    },
    {
        "icon": "⚖️",
        "title": "MAE-Weighted Ensemble",
        "subtitle": "fixed error-based weighting",
        "what": "The base models are combined using inverse-MAE weights, so lower-error models receive higher influence.",
        "why": "This provides the report-style ensemble baseline and gives a simple comparison against the learned hybrid model.",
        "report": "This exactly implements the report's MAE-weighted voting idea.",
    },
    {
        "icon": "🧠",
        "title": "Stacking Hybrid Model",
        "subtitle": "Ridge meta-learner trained on base model predictions",
        "what": "CatBoost and LightGBM predictions become inputs to a Ridge Regression meta-learner.",
        "why": "Instead of using only fixed weights, stacking lets the final model learn how to combine base predictions from data.",
        "report": "This is an extension of the report, not a mismatch. It upgrades the same ensemble idea into a trained hybrid model.",
    },
    {
        "icon": "📏",
        "title": "Evaluation",
        "subtitle": "MAE, MSE, RMSE, R2",
        "what": "Each model is evaluated using MAE, MSE, RMSE, and R2 on the validation period.",
        "why": "These metrics show error size, squared-error penalty, interpretable RMSE, and overall goodness of fit.",
        "report": "The same metric family appears in the implementation, so results can be defended against the report.",
    },
    {
        "icon": "🚀",
        "title": "Deployment",
        "subtitle": "saved joblib model loaded into Streamlit UI",
        "what": "The trained hybrid artifact is saved as joblib and loaded by the Streamlit prediction interface.",
        "why": "A UI turns the notebook work into a presentable application where users can explain and test the model.",
        "report": "Deployment was not the report's main focus, so the Streamlit dashboard adds project value.",
    },
]


TRAINING_STEPS = [
    "Load dataset",
    "Merge sales, stores, and external features",
    "Clean and inspect data",
    "Create calendar and seasonality features",
    "Add Indian festival and monsoon features",
    "Create lag and rolling sales features",
    "Split data using time-based validation",
    "Train CatBoost",
    "Train LightGBM",
    "Generate base model predictions",
    "Create MAE-weighted ensemble",
    "Train Ridge meta-learner for stacking hybrid model",
    "Evaluate all models",
    "Save final model as joblib",
    "Load model in Streamlit for live prediction",
]


REPORT_COMPARISON = [
    ("Dataset: Walmart sales dataset", "Kaggle Walmart dataset", "Matched"),
    ("Models: XGBoost, LightGBM, CatBoost", "CatBoost + LightGBM", "Partially matched"),
    ("Ensemble: MAE-weighted voting", "MAE-weighted voting implemented", "Matched"),
    ("Hybrid Extension: Not deeply trained", "Stacking meta-learner added", "Improved"),
    ("Evaluation: MAE, MSE, RMSE, R2", "Same metrics used", "Matched"),
    ("Deployment: Not covered", "Streamlit UI added", "Added value"),
]


def source_label(filename: str) -> str:
    return f"source/{filename}"


def apply_css(presentation_mode: bool) -> None:
    presentation_css = ""
    if presentation_mode:
        presentation_css = """
        .stMarkdown p, .stMarkdown li, .stExpander p, .stExpander li {
            font-size: 1.08rem !important;
            line-height: 1.7 !important;
        }
        h1 { font-size: 3.2rem !important; }
        h2 { font-size: 2.1rem !important; }
        h3 { font-size: 1.55rem !important; }
        """

    st.markdown(
        f"""
        <style>
        :root {{
            --navy: #17202f;
            --ink: #223047;
            --muted: #5f6c7b;
            --blue: #2563eb;
            --cyan: #0891b2;
            --green: #16a34a;
            --amber: #d97706;
            --red: #dc2626;
            --panel: #ffffff;
            --soft: #eef6ff;
            --line: #dbe5f0;
            --page: #f5f8fc;
            --input: #ffffff;
        }}

        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background: var(--page) !important;
            color: var(--ink) !important;
        }}

        [data-testid="stHeader"] {{
            background: rgba(245, 248, 252, 0.94) !important;
        }}

        [data-testid="stSidebar"] {{
            background: #ffffff !important;
            border-right: 1px solid var(--line);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--ink) !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            background: #f8fbff !important;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 0.35rem 0.45rem;
            margin-bottom: 0.12rem;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: #eef6ff !important;
            border-color: #bfdbfe;
        }}

        [data-testid="stSidebar"] code {{
            background: #eef6ff !important;
            color: #174ea6 !important;
            border: 1px solid #bfdbfe;
        }}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}

        h1, h2, h3,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        label,
        [data-testid="stWidgetLabel"] p {{
            color: var(--navy);
            letter-spacing: 0;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            color: var(--ink);
        }}

        [data-testid="stForm"],
        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            background: #ffffff !important;
            border: 1px solid var(--line);
            border-radius: 8px;
        }}

        [data-testid="stForm"] {{
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(23, 32, 47, 0.05);
        }}

        [data-testid="stMetric"] {{
            padding: 0.8rem;
        }}

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] div {{
            color: var(--ink) !important;
        }}

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] {{
            background: var(--input) !important;
            color: var(--ink) !important;
            border-color: #cbd5e1 !important;
        }}

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span {{
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }}

        div[data-baseweb="select"] svg,
        div[data-baseweb="checkbox"] svg {{
            color: var(--blue) !important;
            fill: var(--blue) !important;
        }}

        button[kind="primary"] {{
            background: var(--blue) !important;
            border: 1px solid var(--blue) !important;
            color: #ffffff !important;
        }}

        button[kind="secondary"] {{
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            color: var(--ink) !important;
        }}

        [data-testid="stAlert"] {{
            background: #fff7ed !important;
            color: #7c2d12 !important;
            border: 1px solid #fed7aa;
            border-radius: 8px;
        }}

        [data-testid="stAlert"] * {{
            color: inherit !important;
        }}

        .main-hero {{
            border: 1px solid var(--line);
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.13), rgba(8, 145, 178, 0.10)),
                #ffffff;
            padding: 1.45rem 1.55rem;
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(23, 32, 47, 0.08);
            margin-bottom: 1rem;
        }}

        .hero-kicker {{
            color: var(--cyan);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }}

        .hero-title {{
            color: var(--navy);
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.04;
            font-weight: 900;
            margin: 0 0 0.55rem 0;
        }}

        .hero-copy {{
            color: var(--ink);
            font-size: 1.04rem;
            line-height: 1.65;
            max-width: 920px;
            margin: 0;
        }}

        .mini-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.8rem;
            margin: 1rem 0;
        }}

        .metric-card, .info-card, .flow-card, .step-card, .say-box {{
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(23, 32, 47, 0.06);
        }}

        .metric-card {{
            min-height: 116px;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .metric-value {{
            color: var(--navy);
            font-size: 1.65rem;
            line-height: 1.2;
            font-weight: 900;
            margin-top: 0.25rem;
        }}

        .metric-note {{
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.25rem;
        }}

        .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.65rem 0 0.1rem 0;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1d4ed8;
            padding: 0.34rem 0.62rem;
            font-size: 0.84rem;
            font-weight: 800;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 800;
            white-space: nowrap;
        }}

        .badge-green {{ background: #dcfce7; color: #166534; }}
        .badge-blue {{ background: #dbeafe; color: #1d4ed8; }}
        .badge-amber {{ background: #fef3c7; color: #92400e; }}

        .flow-card {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.85rem;
            align-items: start;
            margin: 0.45rem 0;
            border-left: 5px solid var(--blue);
        }}

        .flow-icon {{
            width: 46px;
            height: 46px;
            border-radius: 8px;
            background: #eff6ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.45rem;
        }}

        .flow-title {{
            color: var(--navy);
            font-size: 1.08rem;
            font-weight: 900;
            margin-bottom: 0.1rem;
        }}

        .flow-subtitle {{
            color: var(--muted);
            font-size: 0.9rem;
            font-weight: 700;
        }}

        .arrow {{
            color: var(--cyan);
            text-align: center;
            font-size: 1.45rem;
            font-weight: 900;
            margin: 0.15rem 0;
        }}

        .say-box {{
            background: linear-gradient(90deg, #ecfeff, #ffffff);
            border-left: 5px solid var(--cyan);
            margin: 0.85rem 0 1.15rem 0;
        }}

        .say-title {{
            color: #155e75;
            font-weight: 900;
            margin-bottom: 0.25rem;
        }}

        .say-text {{
            color: var(--ink);
            font-size: 0.98rem;
            line-height: 1.55;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(23, 32, 47, 0.06);
        }}

        .comparison-table th {{
            background: #17202f;
            color: #ffffff;
            text-align: left;
            padding: 0.72rem 0.8rem;
            font-size: 0.86rem;
        }}

        .comparison-table td {{
            border-top: 1px solid var(--line);
            color: var(--ink);
            padding: 0.72rem 0.8rem;
            vertical-align: top;
            font-size: 0.92rem;
        }}

        .step-card {{
            min-height: 112px;
            border-top: 4px solid var(--cyan);
        }}

        .step-num {{
            color: var(--cyan);
            font-weight: 900;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .step-text {{
            color: var(--navy);
            font-weight: 850;
            font-size: 1rem;
            margin-top: 0.35rem;
        }}

        .soft-band {{
            border: 1px solid var(--line);
            background: #f8fbff;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.9rem 0;
        }}

        .summary-slide {{
            border: 1px solid var(--line);
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(8, 145, 178, 0.08)),
                #ffffff;
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 14px 34px rgba(23, 32, 47, 0.08);
            margin-top: 0.8rem;
        }}

        .summary-head {{
            display: grid;
            grid-template-columns: 1.3fr 0.7fr;
            gap: 1rem;
            align-items: stretch;
            margin-bottom: 1rem;
        }}

        .summary-title {{
            color: var(--navy);
            font-size: clamp(1.65rem, 3.2vw, 2.7rem);
            font-weight: 950;
            line-height: 1.08;
            margin-bottom: 0.45rem;
        }}

        .summary-subtitle {{
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 760px;
        }}

        .summary-badge {{
            border-radius: 8px;
            background: var(--navy);
            color: #ffffff;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 130px;
        }}

        .summary-badge small {{
            color: #bfdbfe;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 900;
        }}

        .summary-badge strong {{
            color: #ffffff;
            font-size: 1.45rem;
            margin-top: 0.35rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1rem 0;
        }}

        .summary-card {{
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 8px;
            padding: 0.95rem;
            min-height: 145px;
        }}

        .summary-card h4 {{
            color: var(--cyan);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0 0 0.4rem 0;
        }}

        .summary-card p {{
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0;
        }}

        .summary-footer {{
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #174ea6;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            font-weight: 850;
            line-height: 1.55;
            margin-top: 0.9rem;
        }}

        @media (max-width: 860px) {{
            .summary-head,
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff !important;
            box-shadow: 0 8px 20px rgba(23, 32, 47, 0.05);
            margin-bottom: 0.58rem;
        }}

        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {{
            background: #ffffff !important;
            color: var(--ink) !important;
            border-radius: 8px;
        }}

        div[data-testid="stExpander"] summary p {{
            font-weight: 850;
            color: var(--navy);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid var(--line);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 0.8rem;
            background: #eaf3ff;
            color: var(--ink);
        }}

        .stTabs [data-baseweb="tab"] p {{
            color: var(--ink) !important;
            font-weight: 800;
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--navy) !important;
        }}

        .stTabs [aria-selected="true"] p {{
            color: #ffffff !important;
        }}

        {presentation_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json_file(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_comparison_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_artifact(path_text: str) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(path_text)
    if not path.exists():
        return None, f"Model file missing: {source_label(path.name)}"

    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(path), None
    except Exception as exc:
        return None, f"Could not load {source_label(path.name)}. Details: {exc}"


def safe_image(filename: str, caption: str | None = None) -> None:
    path = SOURCE_DIR / filename
    if path.exists():
        st.image(source_label(filename), caption=caption, width="stretch")
    else:
        st.warning(f"Missing image: {source_label(filename)}")


def hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <section class="main-hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{copy}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def review_box(text: str) -> None:
    st.markdown(
        f"""
        <div class="say-box">
            <div class="say-title">What to say in review</div>
            <div class="say-text">Say this: {text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pills(items: list[str]) -> None:
    pills = "".join(f'<span class="pill">{item}</span>' for item in items)
    st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = {
        "Matched": "badge-green",
        "Partially matched": "badge-amber",
        "Improved": "badge-blue",
        "Added value": "badge-blue",
    }.get(status, "badge-blue")
    return f'<span class="badge {cls}">{status}</span>'


def report_comparison_table() -> None:
    rows = []
    for report_method, implementation, status in REPORT_COMPARISON:
        rows.append(
            "<tr>"
            f"<td>{report_method}</td>"
            f"<td>{implementation}</td>"
            f"<td>{status_badge(status)}</td>"
            "</tr>"
        )

    st.markdown(
        """
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Report Method</th>
                    <th>Our Implementation</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        + "".join(rows)
        + """
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def flow_card(step: dict[str, str], index: int) -> None:
    st.markdown(
        f"""
        <div class="flow-card">
            <div class="flow-icon">{step["icon"]}</div>
            <div>
                <div class="flow-title">{index}. {step["title"]}</div>
                <div class="flow-subtitle">{step["subtitle"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flowchart(expand_all: bool = False) -> None:
    st.progress(1.0, text="Raw dataset to deployed Streamlit forecasting dashboard")

    for index, step in enumerate(FLOW_STEPS, start=1):
        flow_card(step, index)
        with st.expander(f"Open step {index}: {step['title']}", expanded=expand_all):
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**What was done**")
                st.write(step["what"])
            with cols[1]:
                st.markdown("**Why it was done**")
                st.write(step["why"])
            with cols[2]:
                st.markdown("**Connection to report**")
                st.write(step["report"])

        if index < len(FLOW_STEPS):
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)


def render_training_steps() -> None:
    for row_start in range(0, len(TRAINING_STEPS), 3):
        cols = st.columns(3)
        for offset, step_text in enumerate(TRAINING_STEPS[row_start : row_start + 3]):
            step_number = row_start + offset + 1
            with cols[offset]:
                st.markdown(
                    f"""
                    <div class="step-card">
                        <div class="step-num">Step {step_number}</div>
                        <div class="step-text">{step_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def metrics_dict() -> dict[str, Any]:
    return load_json_file(str(METRICS_PATH))


def meta_dict() -> dict[str, Any]:
    return load_json_file(str(META_PATH))


def comparison_df() -> pd.DataFrame:
    return load_comparison_csv(str(COMPARISON_PATH))


def render_artifact_warnings() -> None:
    expected_files = [
        "hybrid_stacking_model.joblib",
        "metrics.json",
        "meta_coefficients.json",
        "model_comparison_metrics.csv",
        "actual_vs_predicted_hybrid.png",
        "error_distribution_hybrid.png",
        "sales_over_time_hybrid.png",
    ]
    missing = [source_label(name) for name in expected_files if not (SOURCE_DIR / name).exists()]
    if missing:
        st.warning("Missing project files: " + ", ".join(missing))


def page_home(presentation_mode: bool) -> None:
    hero(
        "Hybrid Model UI",
        "Retail Sales Forecasting Defense Dashboard",
        "A Streamlit presentation hub for the research report, methodology, model training pipeline, evaluation results, and live prediction workflow.",
    )

    render_pills(
        [
            "📘 report aligned",
            "🌳 gradient boosting",
            "⚖️ MAE ensemble",
            "🧠 Ridge stacking",
            "🚀 deployed UI",
        ]
    )

    metrics = metrics_dict()
    best_model = "LightGBM"
    if metrics:
        best_model = min(metrics, key=lambda key: metrics[key].get("MAE", float("inf")))

    cols = st.columns(4)
    with cols[0]:
        metric_card("Project Scope", "End-to-end", "Notebook, model, visuals, and UI")
    with cols[1]:
        metric_card("Best MAE Model", best_model.replace("_", " "), "Lower MAE is better")
    with cols[2]:
        metric_card("Hybrid Layer", "Ridge", "Learns model combination")
    with cols[3]:
        metric_card("Deployment", "Streamlit", "Loaded from source/joblib")

    st.markdown("### Presentation Map")
    map_cols = st.columns(3)
    with map_cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>1. Explain alignment</strong><br>
                Show that the implementation follows the report's gradient boosting and MAE ensemble idea.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with map_cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>2. Walk through pipeline</strong><br>
                Move from Kaggle data to feature engineering, training, stacking, evaluation, and deployment.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with map_cols[2]:
        st.markdown(
            """
            <div class="info-card">
                <strong>3. Demonstrate output</strong><br>
                Use results charts and the prediction demo to show that the model is usable.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Key Result Visuals")
    chart_cols = st.columns(3)
    with chart_cols[0]:
        safe_image("actual_vs_predicted_hybrid.png", "Actual vs predicted weekly sales")
    with chart_cols[1]:
        safe_image("error_distribution_hybrid.png", "Hybrid model error distribution")
    with chart_cols[2]:
        safe_image("sales_over_time_hybrid.png", "Actual vs predicted sales over time")

    review_box(
        "This dashboard is not only a predictor. It is a project defense tool that connects the report, methodology, trained hybrid model, metrics, and deployment in one place."
    )
    render_artifact_warnings()


def page_one_slide_summary(presentation_mode: bool) -> None:
    metrics = metrics_dict()
    hybrid = metrics.get("Stacking_Hybrid_Model", {})
    best_model = "LightGBM"
    best_mae = 0.0
    if metrics:
        best_model = min(metrics, key=lambda key: metrics[key].get("MAE", float("inf")))
        best_mae = metrics[best_model].get("MAE", 0.0)

    hybrid_mae = hybrid.get("MAE", 0.0)
    hybrid_r2 = hybrid.get("R2", 0.0)

    st.html(
        f"""
        <section class="summary-slide">
            <div class="summary-head">
                <div>
                    <div class="summary-title">Retail Sales Forecasting Using a Hybrid Stacking Model</div>
                    <div class="summary-subtitle">
                        A complete implementation of the report idea: Kaggle Walmart sales data, gradient boosting base models,
                        MAE-weighted voting baseline, Ridge Regression stacking, evaluation, and Streamlit deployment.
                    </div>
                </div>
                <div class="summary-badge">
                    <small>Final Contribution</small>
                    <strong>Report-aligned hybrid forecasting dashboard</strong>
                </div>
            </div>

            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Problem</h4>
                    <p>Forecast weekly retail sales accurately so store and department-level demand can be understood before future weeks arrive.</p>
                </div>
                <div class="summary-card">
                    <h4>Dataset</h4>
                    <p>Kaggle Walmart dataset using train.csv, stores.csv, and features.csv merged into one modeling table.</p>
                </div>
                <div class="summary-card">
                    <h4>Features</h4>
                    <p>Calendar, holiday, Indian festival, monsoon, lag sales, and rolling sales features capture seasonality and sales memory.</p>
                </div>
                <div class="summary-card">
                    <h4>Models</h4>
                    <p>CatBoost and LightGBM are trained as gradient boosting base learners, matching the report's modeling direction.</p>
                </div>
                <div class="summary-card">
                    <h4>Hybrid Upgrade</h4>
                    <p>The report's MAE-weighted ensemble is implemented, then extended with a Ridge meta-learner for stacking.</p>
                </div>
                <div class="summary-card">
                    <h4>Deployment</h4>
                    <p>The saved joblib model and generated findings are loaded into Streamlit for live prediction and project defense.</p>
                </div>
            </div>

            <div class="summary-footer">
                Key result: best MAE model is {best_model.replace("_", " ")} with MAE {best_mae:,.2f}.
                The stacking hybrid model reports MAE {hybrid_mae:,.2f} and R2 {hybrid_r2:.4f}.
            </div>
        </section>
        """
    )

    st.markdown("### Presentation Closing Line")
    review_box(
        "This project follows the report's gradient boosting and MAE-weighted ensemble approach, then adds a trained stacking hybrid model and deploys it through a Streamlit dashboard for explanation and prediction."
    )


def page_project_walkthrough(presentation_mode: bool) -> None:
    hero(
        "Project Walkthrough",
        "From Research Report to Working Forecasting App",
        "Use this page as the main guided explanation during review. It connects the report's idea to the implemented model and the deployed Streamlit interface.",
    )

    overview_tab, flow_tab, training_tab, review_tab = st.tabs(
        ["Report Fit", "Interactive Flowchart", "Training Steps", "Review Script"]
    )

    with overview_tab:
        st.markdown("### Why the implementation matches the report")
        st.markdown(
            """
            The report proposed retail sales forecasting with gradient boosting models and an MAE-weighted voting ensemble.
            This implementation follows that structure with CatBoost and LightGBM, then extends it with a stacking layer.

            The important defense point is simple: fixed MAE weights are implemented as the report baseline, and the Ridge
            Regression meta-learner is an upgrade that learns the final blend from model predictions.
            """
        )
        report_comparison_table()
        review_box(
            "The implementation follows the report's gradient boosting ensemble idea, then improves it by adding a trained Ridge Regression stacking layer."
        )

    with flow_tab:
        render_flowchart(expand_all=presentation_mode)
        review_box(
            "The full pipeline starts from the Kaggle dataset, builds forecasting features, trains gradient boosting models, compares an MAE-weighted ensemble, and deploys the saved hybrid model in Streamlit."
        )

    with training_tab:
        render_training_steps()
        review_box(
            "The model was trained in a structured sequence: merge data, engineer forecasting features, split by time, train base learners, create ensemble predictions, train the Ridge meta-learner, evaluate, and save the final joblib artifact."
        )

    with review_tab:
        st.markdown(
            """
            <div class="soft-band">
                <strong>Opening:</strong> This project implements retail sales forecasting on the Kaggle Walmart dataset using gradient boosting models.
                The report proposed XGBoost, LightGBM, CatBoost, and MAE-weighted voting. Our implementation keeps that idea and adds stacking.
            </div>
            <div class="soft-band">
                <strong>Methodology:</strong> We merged train, stores, and features data, created calendar, holiday, festival, monsoon, lag, and rolling features, then validated on future 2012 records.
            </div>
            <div class="soft-band">
                <strong>Hybrid contribution:</strong> The Ridge meta-learner takes CatBoost and LightGBM predictions as inputs and learns how to combine them, which is more adaptive than fixed MAE weights.
            </div>
            <div class="soft-band">
                <strong>Deployment:</strong> The trained artifact is saved as <code>source/hybrid_stacking_model.joblib</code> and loaded in Streamlit for live prediction and presentation.
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_report_alignment(presentation_mode: bool) -> None:
    hero(
        "Report Alignment",
        "The Implementation Extends the Report, It Does Not Conflict With It",
        "This page gives a clean explanation of how the report's proposed methodology maps to the current Streamlit project.",
    )

    st.markdown("### Core Alignment")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>What the report proposed</strong><br><br>
                Retail sales forecasting using gradient boosting models such as XGBoost, LightGBM, and CatBoost, followed by MAE-weighted voting.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>What this project implements</strong><br><br>
                CatBoost and LightGBM base learners, MAE-weighted voting baseline, and a Ridge Regression stacking meta-learner.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Extension Logic")
    st.markdown(
        """
        The report's ensemble uses fixed MAE-based weights. That means the final blend is calculated from validation error,
        but the ensemble itself does not learn a new model. This project keeps that baseline and adds a stacking hybrid
        model, where Ridge Regression learns the relationship between base model predictions and actual sales.
        """
    )

    review_box(
        "The report used MAE-weighted voting as the ensemble idea. We implemented that baseline and extended it with stacking, where Ridge Regression learns how much to trust CatBoost and LightGBM predictions."
    )

    st.markdown("### Report vs Implementation")
    report_comparison_table()

    st.markdown("### Defense Notes")
    st.markdown(
        """
        - XGBoost appears in the report, while this implementation focuses on CatBoost and LightGBM due to time and deployment simplicity.
        - The ensemble concept is still matched because MAE-weighted voting is implemented.
        - The hybrid model is an improvement because the final combination is learned, not only assigned from fixed error weights.
        - Deployment is added value beyond the report because the model is accessible through Streamlit.
        """
    )


def page_training_pipeline(presentation_mode: bool) -> None:
    hero(
        "Training Pipeline Flowchart",
        "Interactive Step-by-Step Model Building Pipeline",
        "Expand each stage to explain what was done, why it matters, and how it connects back to the research report.",
    )

    render_flowchart(expand_all=presentation_mode)

    st.markdown("### Numbered Training Steps")
    render_training_steps()

    review_box(
        "This pipeline proves the model was built like a forecasting system: historical data is transformed into model-ready features, future data is held out for validation, and the final trained artifact is deployed."
    )


def page_results_dashboard(presentation_mode: bool) -> None:
    hero(
        "Results Dashboard",
        "Model Performance and Visual Evidence",
        "Compare CatBoost, LightGBM, MAE-weighted ensemble, and the stacking hybrid model using the saved project metrics and generated plots.",
    )

    metrics = metrics_dict()
    df = comparison_df()

    if metrics:
        best_mae_model = min(metrics, key=lambda key: metrics[key].get("MAE", float("inf")))
        best_r2_model = max(metrics, key=lambda key: metrics[key].get("R2", float("-inf")))
        hybrid = metrics.get("Stacking_Hybrid_Model", {})

        cols = st.columns(4)
        with cols[0]:
            metric_card("Best MAE", best_mae_model.replace("_", " "), f"{metrics[best_mae_model]['MAE']:.2f}")
        with cols[1]:
            metric_card("Best R2", best_r2_model.replace("_", " "), f"{metrics[best_r2_model]['R2']:.4f}")
        with cols[2]:
            metric_card("Hybrid MAE", f"{hybrid.get('MAE', 0):.2f}", "Stacking model")
        with cols[3]:
            metric_card("Hybrid RMSE", f"{hybrid.get('RMSE', 0):.2f}", "Root mean squared error")
    else:
        st.warning(f"Missing or unreadable metrics file: {source_label('metrics.json')}")

    st.markdown("### Model Comparison")
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
        metric_cols = [col for col in ["MAE", "RMSE", "R2"] if col in df.columns]
        if metric_cols:
            st.bar_chart(df.set_index("Model")[metric_cols])
    else:
        st.warning(f"Missing or unreadable comparison CSV: {source_label('model_comparison_metrics.csv')}")

    st.markdown("### Validation Plots")
    plot_tabs = st.tabs(["Actual vs Predicted", "Error Distribution", "Sales Over Time"])
    with plot_tabs[0]:
        safe_image("actual_vs_predicted_hybrid.png", "Stacking Hybrid Model: Actual vs Predicted")
    with plot_tabs[1]:
        safe_image("error_distribution_hybrid.png", "Stacking Hybrid Model: Error Distribution")
    with plot_tabs[2]:
        safe_image("sales_over_time_hybrid.png", "Stacking Hybrid Model: Sales Over Time")

    review_box(
        "These results show that the gradient boosting models perform strongly, the MAE-weighted ensemble baseline is available, and the stacking hybrid model is evaluated with the same report metrics."
    )


def page_hybrid_explanation(presentation_mode: bool) -> None:
    hero(
        "Hybrid Model Explanation",
        "From Fixed Voting to Learned Stacking",
        "This page explains the model architecture in reviewer-friendly language.",
    )

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Base Learner 1</strong><br><br>
                CatBoost learns from categorical store, department, type, festival, and numeric sales-history features.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Base Learner 2</strong><br><br>
                LightGBM learns a fast gradient boosting model from encoded categorical and numeric features.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Meta Learner</strong><br><br>
                Ridge Regression takes the two base predictions and learns the final hybrid forecast.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Architecture")
    st.code(
        """
Input features
    -> CatBoost prediction
    -> LightGBM prediction
        -> Ridge Regression meta-learner
            -> Final weekly sales forecast
        """.strip(),
        language="text",
    )

    meta = meta_dict()
    if meta:
        st.markdown("### Learned Meta-Learner Coefficients")
        coef_cols = st.columns(3)
        with coef_cols[0]:
            metric_card(
                "CatBoost coefficient",
                f"{meta.get('CatBoost_prediction_coefficient', 0):.4f}",
                "Base prediction weight learned by Ridge",
            )
        with coef_cols[1]:
            metric_card(
                "LightGBM coefficient",
                f"{meta.get('LightGBM_prediction_coefficient', 0):.4f}",
                "Base prediction weight learned by Ridge",
            )
        with coef_cols[2]:
            metric_card("Intercept", f"{meta.get('Intercept', 0):.4f}", "Ridge adjustment term")
    else:
        st.warning(f"Missing or unreadable coefficient file: {source_label('meta_coefficients.json')}")

    st.markdown("### MAE Weighted Ensemble vs Stacking")
    compare_cols = st.columns(2)
    with compare_cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>MAE-weighted voting</strong><br><br>
                Uses fixed inverse-error weights. It is simple, explainable, and directly matches the report baseline.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with compare_cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Stacking hybrid model</strong><br><br>
                Trains Ridge Regression on base model predictions, so the final blend is learned from data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    review_box(
        "We extended the report's ensemble method by training a stacking-based hybrid model where Ridge Regression learns how to combine CatBoost and LightGBM predictions."
    )


def prediction_form_defaults() -> dict[str, Any]:
    return {
        "Store": 1,
        "Dept": 1,
        "Type": "A",
        "Size": 151315,
        "IsHoliday": 0,
        "Year": 2012,
        "Month": 6,
        "WeekOfYear": 24,
        "IsMonthEnd": 0,
        "Indian_Festival": "None",
        "Is_Indian_Festival": 0,
        "Is_Monsoon": 1,
        "Sales_Lag_1": 22000.0,
        "Sales_Lag_4": 21500.0,
        "Sales_Lag_12": 20800.0,
        "Sales_RollMean_4": 21850.0,
        "Sales_RollStd_12": 1200.0,
    }


def make_prediction(artifact: dict[str, Any], values: dict[str, Any]) -> float:
    feature_cols = artifact.get("feature_cols", list(values.keys()))
    cat_features = artifact.get("cat_features", ["Store", "Dept", "Type", "Indian_Festival"])

    row = pd.DataFrame([{col: values.get(col) for col in feature_cols}])

    for col in cat_features:
        if col in row:
            row[col] = row[col].astype(str)

    cat_pred = artifact["catboost_model"].predict(row)

    row_lgb = row.copy()
    row_lgb[cat_features] = artifact["lightgbm_encoder"].transform(row_lgb[cat_features].astype(str))
    lgb_pred = artifact["lightgbm_model"].predict(row_lgb)

    meta_input = np.column_stack([cat_pred, lgb_pred])
    final_pred = artifact["meta_model"].predict(meta_input)
    return float(final_pred[0])


def page_prediction_demo(presentation_mode: bool) -> None:
    hero(
        "Prediction Demo",
        "Live Forecast From the Saved Hybrid Model",
        "Enter one Store-Dept-week feature set and generate a weekly sales prediction using the saved joblib artifact.",
    )

    artifact, load_error = load_model_artifact(str(MODEL_PATH))
    if load_error:
        st.warning(load_error)
        st.info("The rest of the dashboard still works because model loading is isolated from the presentation pages.")
        return

    defaults = prediction_form_defaults()
    with st.form("prediction_form"):
        top_cols = st.columns(4)
        with top_cols[0]:
            defaults["Store"] = st.number_input("Store", min_value=1, max_value=45, value=defaults["Store"], step=1)
        with top_cols[1]:
            defaults["Dept"] = st.number_input("Dept", min_value=1, max_value=99, value=defaults["Dept"], step=1)
        with top_cols[2]:
            defaults["Type"] = st.selectbox("Store Type", ["A", "B", "C"], index=0)
        with top_cols[3]:
            defaults["Size"] = st.number_input("Store Size", min_value=1, value=defaults["Size"], step=1000)

        date_cols = st.columns(5)
        with date_cols[0]:
            defaults["Year"] = st.number_input("Year", min_value=2010, max_value=2030, value=defaults["Year"], step=1)
        with date_cols[1]:
            defaults["Month"] = st.number_input("Month", min_value=1, max_value=12, value=defaults["Month"], step=1)
        with date_cols[2]:
            defaults["WeekOfYear"] = st.number_input("Week of Year", min_value=1, max_value=53, value=defaults["WeekOfYear"], step=1)
        with date_cols[3]:
            defaults["IsHoliday"] = int(st.checkbox("Holiday Week", value=bool(defaults["IsHoliday"])))
        with date_cols[4]:
            defaults["IsMonthEnd"] = int(st.checkbox("Month End", value=bool(defaults["IsMonthEnd"])))

        context_cols = st.columns(3)
        with context_cols[0]:
            defaults["Indian_Festival"] = st.selectbox("Indian Festival", ["None", "Holi", "Eid", "Diwali", "Christmas"], index=0)
        with context_cols[1]:
            defaults["Is_Indian_Festival"] = int(defaults["Indian_Festival"] != "None")
            st.metric("Festival Flag", defaults["Is_Indian_Festival"])
        with context_cols[2]:
            defaults["Is_Monsoon"] = int(st.checkbox("Monsoon Season", value=bool(defaults["Is_Monsoon"])))

        lag_cols = st.columns(5)
        with lag_cols[0]:
            defaults["Sales_Lag_1"] = st.number_input("Sales Lag 1", min_value=0.0, value=defaults["Sales_Lag_1"], step=500.0)
        with lag_cols[1]:
            defaults["Sales_Lag_4"] = st.number_input("Sales Lag 4", min_value=0.0, value=defaults["Sales_Lag_4"], step=500.0)
        with lag_cols[2]:
            defaults["Sales_Lag_12"] = st.number_input("Sales Lag 12", min_value=0.0, value=defaults["Sales_Lag_12"], step=500.0)
        with lag_cols[3]:
            defaults["Sales_RollMean_4"] = st.number_input("Rolling Mean 4", min_value=0.0, value=defaults["Sales_RollMean_4"], step=500.0)
        with lag_cols[4]:
            defaults["Sales_RollStd_12"] = st.number_input("Rolling Std 12", min_value=0.0, value=defaults["Sales_RollStd_12"], step=100.0)

        submitted = st.form_submit_button("Predict Weekly Sales", type="primary")

    if submitted:
        try:
            prediction = make_prediction(artifact, defaults)
            st.success(f"Predicted weekly sales: {prediction:,.2f}")
        except Exception as exc:
            st.warning(f"Prediction failed, but the app stayed running. Details: {exc}")

    review_box(
        "The final trained model is saved as a joblib artifact and loaded by Streamlit, so the project moves beyond a notebook into an interactive deployment."
    )


def page_about_viva(presentation_mode: bool) -> None:
    hero(
        "About / Viva",
        "Concise Answers for Project Review",
        "Use these points to answer common questions about dataset choice, methodology, model design, and limitations.",
    )

    qa_items = [
        (
            "Why use a time-based split?",
            "Forecasting should imitate reality. The model learns from earlier records and is validated on future 2012 records.",
        ),
        (
            "Why CatBoost?",
            "CatBoost is strong with categorical variables such as Store, Dept, Type, and Indian_Festival.",
        ),
        (
            "Why LightGBM?",
            "LightGBM is fast and accurate for large tabular datasets, making it a useful complementary base learner.",
        ),
        (
            "Why add Indian festival and monsoon features to a Walmart dataset?",
            "They demonstrate context-aware feature engineering. In a local retail setting, festival and seasonal indicators can influence demand.",
        ),
        (
            "How is stacking different from MAE voting?",
            "MAE voting uses fixed weights from validation error. Stacking trains a Ridge Regression meta-learner on base predictions.",
        ),
        (
            "What is one limitation?",
            "A production-grade stacking model should use out-of-fold predictions, ideally with TimeSeriesSplit, to train the meta-learner more rigorously.",
        ),
    ]

    for question, answer in qa_items:
        with st.expander(question, expanded=presentation_mode):
            st.write(answer)

    st.markdown("### Files Used by the UI")
    st.markdown(
        """
        - `source/hybrid_stacking_model.joblib`
        - `source/metrics.json`
        - `source/meta_coefficients.json`
        - `source/model_comparison_metrics.csv`
        - `source/actual_vs_predicted_hybrid.png`
        - `source/error_distribution_hybrid.png`
        - `source/sales_over_time_hybrid.png`
        """
    )

    review_box(
        "My contribution is the full bridge from research methodology to implementation: feature engineering, gradient boosting models, weighted ensemble baseline, stacking hybrid model, evaluation, and Streamlit deployment."
    )


def render_sidebar() -> tuple[str, bool]:
    st.sidebar.title("Forecasting UI")
    st.sidebar.caption("Mini project defense dashboard")
    page = st.sidebar.radio("Navigate", PAGE_OPTIONS, index=0)
    presentation_mode = st.sidebar.checkbox(
        "Presentation Mode",
        value=False,
        help="Increases text emphasis and opens walkthrough details for review explanation.",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Artifact base path**")
    st.sidebar.code("source/")
    return page, presentation_mode


def main() -> None:
    st.set_page_config(
        page_title="Hybrid Sales Forecasting UI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    page, presentation_mode = render_sidebar()
    apply_css(presentation_mode)

    if page == "Home":
        page_home(presentation_mode)
    elif page == "One-Slide Summary":
        page_one_slide_summary(presentation_mode)
    elif page == "Project Walkthrough":
        page_project_walkthrough(presentation_mode)
    elif page == "Report Alignment":
        page_report_alignment(presentation_mode)
    elif page == "Training Pipeline Flowchart":
        page_training_pipeline(presentation_mode)
    elif page == "Results Dashboard":
        page_results_dashboard(presentation_mode)
    elif page == "Hybrid Model Explanation":
        page_hybrid_explanation(presentation_mode)
    elif page == "Prediction Demo":
        page_prediction_demo(presentation_mode)
    elif page == "About / Viva":
        page_about_viva(presentation_mode)


if __name__ == "__main__":
    main()

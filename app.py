import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent_orchestrator import run_agentic_pipeline
from src.db import authenticate_user, ensure_default_user, get_recent_sessions, init_db, register_user, save_session


st.set_page_config(page_title="AgriAgent", page_icon="🌾", layout="wide")


def load_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #f5fbf7;
            --bg-2: #eef8ff;
            --panel: rgba(255,255,255,0.9);
            --panel-strong: #ffffff;
            --primary: #1e7a44;
            --primary-soft: #dff7e8;
            --secondary: #132c42;
            --muted: #5d6b73;
            --line: rgba(19, 44, 66, 0.10);
            --shadow: 0 18px 42px rgba(14, 53, 30, 0.10);
            --success: #2e9d64;
            --warning: #e9b248;
        }

        .stApp {
            background: linear-gradient(135deg, var(--bg-1) 0%, #edf8f0 35%, var(--bg-2) 100%);
            color: var(--secondary);
        }

        .stAppHeader {
            background: transparent;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        .brand {
            font-size: 3rem;
            font-weight: 900;
            letter-spacing: -0.05em;
            color: var(--primary);
            margin: 0.2rem 0 0.25rem 0;
        }

        .sub-brand {
            font-size: 1.08rem;
            color: var(--muted);
            margin-bottom: 1.5rem;
            max-width: 840px;
        }

        .glass-card {
            background: rgba(255,255,255,0.86);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1.15rem 1.25rem;
            backdrop-filter: blur(8px);
        }

        .login-shell {
            max-width: 760px;
            margin: 2rem auto 1rem auto;
            padding: 2.1rem 2.2rem;
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(241,250,245,0.96));
            border: 1px solid rgba(30,122,68,0.12);
            box-shadow: 0 28px 70px rgba(15, 58, 38, 0.12);
        }

        .premium-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(246,252,249,0.9));
            border: 1px solid rgba(18, 55, 35, 0.06);
            border-radius: 24px;
            padding: 1rem 1.2rem 1.4rem 1.2rem;
            box-shadow: var(--shadow);
        }

        .sidebar .block-container {
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111f2d 0%, #19354b 100%);
        }

        .sidebar-content {
            color: white;
        }

        .main .stButton > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, var(--primary), #2da167);
            color: white;
            font-weight: 700;
            padding: 0.7rem 1.1rem;
            box-shadow: 0 12px 25px rgba(30,122,68,0.20);
        }

        .main .stButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            border-radius: 12px 12px 0 0;
            background: rgba(255,255,255,0.6);
            border: 1px solid var(--line);
            color: var(--secondary);
            font-weight: 700;
        }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 18px;
            border: 2px dashed rgba(30,122,68,0.5);
            background: rgba(255,255,255,0.75);
            min-height: 130px;
        }

        .workflow-outer {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin: 1.4rem 0 2rem 0;
        }

        .workflow-box {
            width: min(470px, 78vw);
            min-height: 72px;
            border: 3px solid rgba(26, 107, 153, 0.8);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-weight: 800;
            color: var(--secondary);
            box-shadow: 0 12px 28px rgba(26, 107, 153, 0.10);
            padding: 0.8rem 1rem;
            white-space: pre-line;
            line-height: 1.2;
            background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(255,255,255,0.55));
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(230,247,238,0.96));
            border: 1px solid rgba(30,122,68,0.12);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 12px 22px rgba(23, 66, 79, 0.08);
            min-height: 92px;
        }

        .pill-nav {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin: 0.4rem 0 1rem 0;
        }

        .pill-nav span {
            background: rgba(255,255,255,0.8);
            color: var(--secondary);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.5rem 0.9rem;
            font-weight: 700;
            box-shadow: 0 6px 14px rgba(14, 53, 30, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.read_csv("data/sample_agri_dataset.csv")

    file_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(io.BytesIO(file_bytes))
    if suffix == ".json":
        return pd.read_json(io.BytesIO(file_bytes))
    raise ValueError("Unsupported file format. Please upload CSV, JSON, or Excel data.")


def read_uploaded_files(uploaded_files) -> pd.DataFrame:
    if not uploaded_files:
        return pd.read_csv("data/sample_agri_dataset.csv")

    frames = []
    for uploaded_file in uploaded_files:
        frames.append(read_uploaded_file(uploaded_file))

    if not frames:
        return pd.read_csv("data/sample_agri_dataset.csv")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.reset_index(drop=True)
    return merged


def render_auth_page() -> None:
    load_css()
    st.markdown('<div class="brand">AgriAgent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-brand">Agentic AI Platform for Automated Data Engineering and Intelligent Analytics</div>', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Login", "Create account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            if not username or not password:
                st.error("Username and password are required.")
            else:
                user = authenticate_user(username, password)
                if user:
                    st.session_state["username"] = user
                    st.rerun()
                else:
                    st.error("Invalid credentials. Use the default admin/admin123 account or create a new user.")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose username")
            email = st.text_input("Email (optional)")
            new_password = st.text_input("Password", type="password")
            submit_register = st.form_submit_button("Create account")

        if submit_register:
            ok, message = register_user(new_username, new_password, email)
            if ok:
                st.success(message)
                st.session_state["username"] = new_username.strip()
                st.rerun()
            else:
                st.error(message)

    st.markdown("""
    <div class="login-shell">
      <h3 style="margin-top:0; color:#1f7a45;">Default account</h3>
      <p style="margin: 0.45rem 0; font-size: 1rem;"><strong>Username:</strong> admin</p>
      <p style="margin: 0.25rem 0 0 0; font-size: 1rem;"><strong>Password:</strong> admin123</p>
    </div>
    """, unsafe_allow_html=True)


def render_workflow_flow() -> None:
    stages = [
        ("USER", "User"),
        ("UPLOAD", "Upload Agriculture Data\nCSV / JSON / Excel"),
        ("PROFILE", "Data Profiler\nSoil • Crop • Weather • Yield • Missing values"),
        ("SCHEMA", "LLM Schema Agent\nSemantic Mapping"),
        ("BRONZE", "Bronze\nRaw Agriculture Data"),
        ("ETL", "LLM ETL Agent\nSQL / Python"),
        ("VALIDATION", "Validation Agent"),
        ("FIX", "LLM Fix"),
        ("EXECUTE", "Execute"),
        ("SILVER", "Silver\nClean Agriculture Data"),
        ("EDA", "Automated EDA"),
        ("GOLD", "Gold\nML/Analytics Ready Data"),
        ("CROP", "Crop Analysis\nHealth • Yield • Trends"),
        ("MODEL", "ML Model\nCrop / Yield Prediction"),
        ("TEXT", "Text-to-Analytics"),
        ("DASHBOARD", "Dashboard"),
    ]

    html = "<div class='workflow-outer'>"
    for idx, (key, label) in enumerate(stages):
        color = [
            '#dff6e8', '#dfeeff', '#edf3ff', '#f1e8ff', '#f3e4d3', '#dfeaff', '#e0f4ea', '#fecaca', '#e9d5ff', '#dde5ec', '#d5f5e3', '#f8e8c7', '#f5e1ff', '#defbe8', '#ffebeb', '#dbeafe'
        ][idx % 16]
        html += f"<div class='workflow-box' style='background:{color};'>{key}<br>{label}</div>"
    html += "</div>"

    st.markdown(
        """
        <style>
        .workflow-outer {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin: 1rem 0 2rem 0;
        }
        .workflow-box {
            width: min(420px, 70vw);
            min-height: 68px;
            border: 3px solid #1f7a8c;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-weight: 800;
            color: #1d3557;
            box-shadow: 0 8px 20px rgba(31, 122, 140, 0.12);
            padding: 0.8rem;
            white-space: pre-line;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(html, unsafe_allow_html=True)


def render_dashboard() -> None:
    load_css()
    st.sidebar.markdown('<div class="sidebar-content"><h3 style="color:#ffffff; margin-bottom:0.2rem;">Welcome</h3><p style="color:#dfeef6; margin:0 0 1rem 0;">%s</p></div>' % st.session_state['username'], unsafe_allow_html=True)
    st.sidebar.button("Logout", on_click=lambda: st.session_state.pop("username", None))

    recent_sessions = get_recent_sessions(st.session_state["username"], limit=5)
    if recent_sessions:
        st.sidebar.markdown('<div style="color:#dfeef6; font-weight:700; margin-top:1rem;">Recent sessions</div>', unsafe_allow_html=True)
        for item in recent_sessions:
            st.sidebar.caption(f"{item['file_name']} • {item['created_at']}")

    st.title("AgriAgent")
    st.caption("Agentic AI Platform for Automated Data Engineering and Intelligent Analytics")

    st.markdown(
        '<div class="pill-nav"><span>Overview</span><span>Schema Match</span><span>ETL + Validation</span><span>Analytics</span><span>Dashboard</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 1.2rem; color: #2c6e49; font-weight: 600;">
        This workflow understands agriculture-oriented datasets by profiling schema, matching fields to agronomic concepts, cleaning and transforming data, then generating analytics-ready outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_workflow_flow()

    uploaded_files = st.file_uploader("Upload one or more agricultural datasets", type=["csv", "xlsx", "xls", "json"], accept_multiple_files=True)

    if st.button("Run AgriAgent pipeline"):
        try:
            raw_df = read_uploaded_files(uploaded_files)
            report = run_agentic_pipeline(raw_df)

            st.success("Pipeline executed successfully.")

            file_names = ", ".join(f.name for f in uploaded_files) if uploaded_files else "sample_agri_dataset.csv"
            summary_text = (
                f"Rows={report['summary'].get('rows', 0)}, "
                f"Columns={report['summary'].get('columns', 0)}, "
                f"Domain={report['workflow'].get('domain_detected', 'unknown')}"
            )
            save_session(st.session_state["username"], file_names, summary_text)

            summary = report["summary"]
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown('<div class="metric-card"><div style="color:#5f6f72; font-size:0.8rem;">Rows</div><div style="font-size:2rem; font-weight:800; color:#1d3557;">%s</div></div>' % summary.get("rows", 0), unsafe_allow_html=True)
            col2.markdown('<div class="metric-card"><div style="color:#5f6f72; font-size:0.8rem;">Columns</div><div style="font-size:2rem; font-weight:800; color:#1d3557;">%s</div></div>' % summary.get("columns", 0), unsafe_allow_html=True)
            col3.markdown('<div class="metric-card"><div style="color:#5f6f72; font-size:0.8rem;">Domain</div><div style="font-size:1.4rem; font-weight:800; color:#1d3557;">%s</div></div>' % report["workflow"].get("domain_detected", "unknown"), unsafe_allow_html=True)
            col4.markdown('<div class="metric-card"><div style="color:#5f6f72; font-size:0.8rem;">Mapped fields</div><div style="font-size:2rem; font-weight:800; color:#1d3557;">%s</div></div>' % len(report["schema_mapping"]), unsafe_allow_html=True)

            st.subheader("Detected agriculture context")
            st.json(report["workflow"])

            st.subheader("Data profiling")
            st.dataframe(pd.DataFrame(report["profile_report"]), width="stretch")

            st.subheader("Schema mapping")
            mapping_df = pd.DataFrame(report["schema_mapping"])
            if mapping_df.empty:
                st.info("No strong agriculture schema matches were detected. Review the raw columns manually and tune the matching rules.")
            else:
                st.dataframe(mapping_df, width="stretch")

            st.subheader("ETL execution flow")
            for step in report["workflow"]["etl_steps"]:
                st.write(f"- {step}")

            st.subheader("Data quality validation")
            validation = report["validation"]
            st.json(validation)

            st.subheader("Agriculture insights")
            for insight in report["insights"]:
                st.write(f"- {insight}")

            st.subheader("Bronze vs Silver vs Gold data")
            bronze_tab, silver_tab, gold_tab = st.tabs(["Bronze", "Silver", "Gold"])
            with bronze_tab:
                st.dataframe(report["bronze"].head(10), width="stretch")
            with silver_tab:
                st.dataframe(report["silver"].head(10), width="stretch")
            with gold_tab:
                st.dataframe(report["gold"].head(10), width="stretch")

            st.subheader("Analytics charts")
            for _, fig in report["charts"].items():
                st.plotly_chart(fig, width="stretch")

            gold_csv = report["gold"].to_csv(index=False)
            st.download_button(
                label="Download Gold layer CSV",
                data=gold_csv,
                file_name="agriagent_gold_dataset.csv",
                mime="text/csv",
            )

        except Exception as exc:  # pragma: no cover
            st.error(f"The pipeline failed: {exc}")

    else:
        st.info("Use the sample dataset or upload a new file to start the pipeline.")
        sample_df = pd.read_csv("data/sample_agri_dataset.csv")
        st.dataframe(sample_df.head(10), width="stretch")


init_db()
ensure_default_user()

if "username" not in st.session_state or not st.session_state["username"]:
    render_auth_page()
else:
    render_dashboard()

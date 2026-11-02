import io
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception:  # pragma: no cover
    colors = None
    letter = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None

from src.agent_orchestrator import DataCleaningAgent, run_agentic_pipeline
from src.data_pipeline import (
    evaluate_crop_thresholds,
    predict_crop_suitability,
    predict_yield_for_crop,
    recommend_best_crops_for_region,
)
from src.db import authenticate_user, ensure_default_user, get_recent_sessions, init_db, register_user, save_session
import plotly.io as pio

from src.llm import generate_agronomy_explanation, generate_professional_pdf_copy


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


def _build_minimal_pdf(lines: list[str]) -> bytes:
    safe_lines = []
    for line in lines:
        clean_line = str(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        safe_lines.append(clean_line)

    content_parts = []
    y = 760
    for line in safe_lines:
        content_parts.append(f"BT /F1 12 Tf 50 {y} Td ({line}) Tj ET")
        y -= 18
    content_stream = "\n".join(content_parts).encode("latin-1", errors="replace")

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content_stream)} >>\nstream\n{content_stream.decode('latin-1')}\nendstream",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents 4 0 R >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n{obj}\nendobj\n".encode("latin-1", errors="replace"))

    xref_position = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n".encode("latin-1"))
    return bytes(pdf)


def _figure_to_png_bytes(fig) -> bytes:
    if fig is None:
        raise ValueError("Chart figure is missing.")

    for func in (
        lambda: pio.to_image(fig, format="png", width=900, height=420, scale=2),
        lambda: pio.to_image(fig, format="png", engine="kaleido", scale=2),
    ):
        try:
            image_bytes = func()
            if isinstance(image_bytes, (bytes, bytearray)) and image_bytes:
                return bytes(image_bytes)
        except Exception:
            continue

    buffer = io.BytesIO()
    try:
        pio.write_image(fig, buffer, format="png", scale=2)
        image_bytes = buffer.getvalue()
        if image_bytes:
            return image_bytes
    except Exception:
        pass

    try:
        plt.close("all")
        placeholder_fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "Chart preview unavailable\nin deployed environment",
            ha="center",
            va="center",
            fontsize=12,
            color="#123a2a",
            wrap=True,
        )
        placeholder_fig.tight_layout()
        placeholder_fig.savefig(buffer, format="png", dpi=150)
        image_bytes = buffer.getvalue()
        if image_bytes:
            return image_bytes
    except Exception:
        pass

    raise ValueError("Unable to render chart PNG for the PDF export.")


def generate_crop_report_pdf(
    crop_name: str,
    result: dict,
    yield_forecast: dict | None,
    threshold_review: dict | None,
    recommended_crops: list[dict],
    llm_explanation: dict | None = None,
    selected_region: str | None = None,
) -> bytes:
    yield_value = yield_forecast.get("predicted_yield_ton_per_hectare") if isinstance(yield_forecast, dict) else None
    status = str(result.get("suitability_label", "N/A"))
    confidence = float(result.get("confidence", 0.0) or 0.0)
    reasons = result.get("reasons", [])[:6]
    recommendations = []
    next_actions = []
    if llm_explanation:
        recommendations = llm_explanation.get("recommendations", [])[:5]
        next_actions = llm_explanation.get("next_actions", [])[:5]

    region_label = str(selected_region).strip() if selected_region and str(selected_region).strip() and str(selected_region).strip() != "Any" else "Selected region"
    professional_copy = generate_professional_pdf_copy(
        crop_name,
        status,
        confidence,
        yield_value,
        reasons,
        recommendations,
        next_actions,
        district=region_label,
    )

    report_lines = [
        professional_copy.get("headline", "AgriAgent Crop Suitability Report"),
        f"Region: {region_label}",
        f"Crop: {crop_name}",
        f"Status: {status}",
        f"Confidence: {confidence * 100:.1f}%",
        f"Suitable: {'Yes' if result.get('suitable') else 'No'}",
    ]

    if yield_value is not None:
        report_lines.append(f"Estimated yield: {yield_value:.2f} tons/hectare")

    if professional_copy.get("key_points"):
        report_lines.append("Key findings:")
        report_lines.extend(f"- {point}" for point in professional_copy["key_points"][:6])

    if threshold_review and threshold_review.get("checks"):
        report_lines.append("Threshold review:")
        for check in threshold_review.get("checks", [])[:8]:
            metric = check.get("metric", "")
            status_value = check.get("status", "")
            value = check.get("value")
            target = check.get("target")
            report_lines.append(f"- {metric}: {value} | target={target} | status={status_value}")

    if llm_explanation:
        report_lines.append("AI agronomy explanation:")
        report_lines.append(f"Summary: {llm_explanation.get('summary', '')}")
        for risk in llm_explanation.get("risks", [])[:5]:
            report_lines.append(f"Risk: {risk}")
        for recommendation in recommendations:
            report_lines.append(f"Recommendation: {recommendation}")
        for action in next_actions:
            report_lines.append(f"Next action: {action}")

    if recommended_crops:
        report_lines.append("Regional recommendations:")
        for item in recommended_crops[:5]:
            crop = item.get("crop", "")
            score = item.get("score", 0)
            report_lines.append(f"- {crop}: score={score}")

    try:
        if not all([colors, letter, getSampleStyleSheet, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle]):
            raise RuntimeError("reportlab is required to generate PDF exports.")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=42, leftMargin=42, topMargin=30, bottomMargin=30)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0e5b3d"),
            spaceAfter=10,
            borderPadding=0,
        )
        subtitle_style = ParagraphStyle(
            "SubtitleStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#3f5d52"),
            spaceAfter=10,
        )
        section_style = ParagraphStyle(
            "SectionStyle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1d3f2d"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1f2d24"),
            spaceAfter=4,
        )
        summary_style = ParagraphStyle(
            "SummaryStyle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#25372e"),
            backColor=colors.HexColor("#f3faf5"),
            borderColor=colors.HexColor("#d2ead9"),
            borderWidth=1,
            spaceAfter=12,
            leftIndent=8,
            rightIndent=8,
            paddingLeft=8,
            paddingRight=8,
            paddingTop=6,
            paddingBottom=6,
        )

        story = []
        story.append(Paragraph("AgriAgent", ParagraphStyle("Brand", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#2d7d57"), spaceAfter=0)))
        headline = professional_copy.get("headline", "Crop Suitability Report")
        if region_label != "Selected region":
            headline = f"{headline} — {region_label}"
        story.append(Paragraph(headline, title_style))
        story.append(Paragraph(f"Prepared for {crop_name} evaluation in {region_label}", subtitle_style))

        metric_data = [
            ["Crop", crop_name],
            ["Status", status],
            ["Confidence", f"{confidence * 100:.1f}%"],
            ["Estimated yield", f"{yield_value:.2f} tons/ha" if yield_value is not None else "Not available"],
        ]
        metric_table = Table(metric_data, colWidths=[165, 350])
        metric_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e7a44")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#d9e9df")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f5fbf7")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(metric_table)
        story.append(Spacer(1, 8))

        story.append(Paragraph("Executive summary", section_style))
        story.append(Paragraph(professional_copy.get("summary", "Crop suitability assessment summary is available."), summary_style))

        if professional_copy.get("key_points"):
            story.append(Paragraph("Key findings", section_style))
            for point in professional_copy["key_points"][:5]:
                story.append(Paragraph(f"- {point}", body_style))

        if llm_explanation:
            story.append(Paragraph("AI agronomy explanation", section_style))
            story.append(Paragraph(f"Summary: {llm_explanation.get('summary', 'No summary available.')}", body_style))
            for risk in llm_explanation.get("risks", [])[:5]:
                story.append(Paragraph(f"Risk: {risk}", body_style))
            for recommendation in recommendations:
                story.append(Paragraph(f"Recommendation: {recommendation}", body_style))
            for action in next_actions:
                story.append(Paragraph(f"Next action: {action}", body_style))

        if professional_copy.get("recommendations"):
            story.append(Paragraph("Recommendations", section_style))
            for item in professional_copy["recommendations"][:5]:
                story.append(Paragraph(f"- {item}", body_style))

        if professional_copy.get("next_steps"):
            story.append(Paragraph("Next steps", section_style))
            for item in professional_copy["next_steps"][:5]:
                story.append(Paragraph(f"- {item}", body_style))

        if threshold_review and threshold_review.get("checks"):
            story.append(Paragraph("Threshold review", section_style))
            for check in threshold_review.get("checks", [])[:8]:
                metric = check.get("metric", "")
                status_value = check.get("status", "")
                value = check.get("value")
                target = check.get("target")
                story.append(Paragraph(f"- {metric}: {value} | target={target} | status={status_value}", body_style))

        if recommended_crops:
            story.append(Paragraph("Regional recommendations", section_style))
            for item in recommended_crops[:5]:
                crop = item.get("crop", "")
                score = item.get("score", 0)
                story.append(Paragraph(f"- {crop}: score={score}", body_style))

        doc.build(story)
        return buffer.getvalue()
    except Exception:
        return _build_minimal_pdf(report_lines)


def generate_dashboard_report_pdf(report: dict) -> bytes:
    summary = report.get("summary", {})
    workflow = report.get("workflow", {})
    insights = report.get("insights", [])
    charts = report.get("charts", {})

    try:
        if not all([colors, letter, getSampleStyleSheet, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image, PageBreak]):
            raise RuntimeError("reportlab is required to generate PDF exports.")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=42, leftMargin=42, topMargin=30, bottomMargin=30)
        styles = getSampleStyleSheet()

        brand_style = ParagraphStyle(
            "BrandStyle",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.HexColor("#2d7d57"),
            spaceAfter=0,
        )
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#113a2c"),
            spaceAfter=8,
        )
        subtitle_style = ParagraphStyle(
            "SubtitleStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#44635a"),
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "SectionStyle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1d3f2d"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1f2d24"),
            spaceAfter=4,
        )
        summary_box = ParagraphStyle(
            "SummaryBox",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#123a2a"),
            backColor=colors.HexColor("#eefaf2"),
            borderColor=colors.HexColor("#cfead8"),
            borderWidth=1,
            spaceBefore=6,
            spaceAfter=12,
            leftIndent=8,
            rightIndent=8,
            paddingLeft=8,
            paddingRight=8,
            paddingTop=7,
            paddingBottom=7,
        )

        story = []
        story.append(Paragraph("AgriAgent", brand_style))
        story.append(Paragraph("Dashboard Report", title_style))
        story.append(Paragraph("Agriculture intelligence overview and operational dashboard", subtitle_style))

        metric_rows = [
            ["Domain", workflow.get("domain_detected", "unknown")],
            ["Rows", str(summary.get("rows", 0))],
            ["Columns", str(summary.get("columns", 0))],
            ["Mapped fields", str(len(report.get("schema_mapping", [])))],
            ["Validation status", str(report.get("validation", {}).get("status", "unknown"))],
        ]
        metrics = Table(metric_rows, colWidths=[180, 320])
        metrics.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e7a44")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#d9e9df")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f5fbf7")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(metrics)
        story.append(Spacer(1, 14))

        story.append(Paragraph("Executive summary", section_style))
        exec_text = (
            f"The current dataset covers {summary.get('rows', 0)} rows and {summary.get('columns', 0)} columns across the {workflow.get('domain_detected', 'unknown')} domain. "
            f"The pipeline mapped {len(report.get('schema_mapping', []))} agronomic fields and validated the transformation workflow for export-ready analysis."
        )
        story.append(Paragraph(exec_text, summary_box))

        story.append(Paragraph("Key insights", section_style))
        for insight in insights[:8]:
            story.append(Paragraph(f"- {insight}", body_style))

        if workflow.get("etl_steps"):
            story.append(Paragraph("ETL flow", section_style))
            for step in workflow["etl_steps"][:8]:
                story.append(Paragraph(f"- {step}", body_style))

        story.append(PageBreak())

        if charts:
            story.append(Paragraph("Analytics charts", section_style))
            chart_items = list(charts.items())[:4]
            for index, (chart_name, fig) in enumerate(chart_items):
                try:
                    png_bytes = _figure_to_png_bytes(fig)
                    story.append(Paragraph(str(chart_name), body_style))
                    story.append(Image(io.BytesIO(png_bytes), width=500, height=260))
                    if index < len(chart_items) - 1:
                        story.append(Spacer(1, 12))
                except Exception:
                    story.append(Paragraph(f"Chart unavailable: {chart_name}", body_style))

        doc.build(story)
        return buffer.getvalue()
    except Exception:
        lines = [
            "AgriAgent Dashboard Report",
            f"Domain: {workflow.get('domain_detected', 'unknown')}",
            f"Rows: {summary.get('rows', 0)}",
            f"Columns: {summary.get('columns', 0)}",
        ]
        for insight in insights[:8]:
            lines.append(f"- {insight}")
        return _build_minimal_pdf(lines)


def read_uploaded_file(uploaded_file, max_rows: int | None = None) -> pd.DataFrame:
    if uploaded_file is None:
        df = read_default_dataset_folder()
        if max_rows is not None:
            return df.head(max_rows).reset_index(drop=True)
        return df.reset_index(drop=True)

    file_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.head(max_rows).reset_index(drop=True) if max_rows is not None else df.reset_index(drop=True)
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df.head(max_rows).reset_index(drop=True) if max_rows is not None else df.reset_index(drop=True)
    if suffix == ".json":
        df = pd.read_json(io.BytesIO(file_bytes))
        return df.head(max_rows).reset_index(drop=True) if max_rows is not None else df.reset_index(drop=True)
    raise ValueError("Unsupported file format. Please upload CSV, JSON, or Excel data.")


def limit_dataframe_for_quick_analysis(df: pd.DataFrame, max_rows: int = 20000) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def read_default_dataset_folder() -> pd.DataFrame:
    data_dir = Path("data")
    if not data_dir.exists():
        return pd.DataFrame()

    files = sorted(data_dir.glob("*"))
    default_files = [p for p in files if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}]
    if not default_files:
        return pd.DataFrame()

    frames = []
    for path in default_files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            elif path.suffix.lower() == ".json":
                df = pd.read_json(path)
            else:
                continue
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged.reset_index(drop=True)


def read_uploaded_files(uploaded_files) -> pd.DataFrame:
    if uploaded_files is None:
        return read_default_dataset_folder()

    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    files_to_read = [item for item in uploaded_files if item is not None]
    if not files_to_read:
        return read_default_dataset_folder()

    frames = [read_uploaded_file(item) for item in files_to_read]
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return read_default_dataset_folder()

    merged = pd.concat(valid_frames, ignore_index=True, sort=False)
    merged = merged.reset_index(drop=True)
    return merged


def clean_uploaded_data(uploaded_files) -> pd.DataFrame:
    raw_df = read_uploaded_files(uploaded_files)
    cleaned = DataCleaningAgent.clean(raw_df)
    if isinstance(cleaned, list):
        if not cleaned:
            return read_default_dataset_folder()
        return pd.concat(cleaned, ignore_index=True, sort=False).reset_index(drop=True)
    if cleaned.empty:
        return read_default_dataset_folder()
    return cleaned.reset_index(drop=True)


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


def render_crop_suitability_checker(training_df: pd.DataFrame) -> None:
    st.subheader("Crop suitability checker")

    def clean_location_choices(values):
        cleaned = []
        for value in values:
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none", "null"}:
                continue
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
                continue
            cleaned.append(text)
        return sorted(set(cleaned))

    column_names = {str(col).strip().lower(): col for col in training_df.columns}
    crop_col = column_names.get("crop") or next((col for col in training_df.columns if str(col).strip().lower() == "crop"), None)

    location_column_keywords = ("district", "region", "state", "province", "location", "block", "taluka", "city", "area", "village")
    location_cols = []
    for col in training_df.columns:
        lowered = str(col).strip().lower()
        if lowered in {"district", "district_name", "region", "region_name", "state", "state_name", "province", "location", "location_name", "city", "block", "taluka", "area", "area_name", "village"}:
            location_cols.append(col)
            continue
        if any(keyword in lowered for keyword in location_column_keywords):
            location_cols.append(col)

    crop_values = []
    if crop_col is not None:
        crop_values = training_df[crop_col].dropna().astype(str).unique().tolist()
    if not crop_values:
        crop_values = ["Rice", "Wheat", "Cotton", "Soybean", "Maize", "Sugarcane", "Groundnut", "Grapes"]
    crop_options = sorted({str(value).strip() for value in crop_values if str(value).strip()})

    location_values = []
    for col in location_cols:
        location_values.extend(training_df[col].dropna().astype(str).unique().tolist())

    district_options = ["Any"] + clean_location_choices(location_values)

    with st.form("crop_suitability_form"):
        crop_name = st.selectbox("Crop to test", crop_options)
        district = st.selectbox("District / Region (optional)", district_options)
        climate = st.selectbox("Climate profile (optional)", ["Any", "Humid", "Semi-arid", "Temperate", "Tropical", "Dry"])

        col1, col2, col3 = st.columns(3)
        soil_ph = col1.number_input("Soil pH", min_value=3.0, max_value=9.5, value=6.5, step=0.1)
        rainfall_mm = col2.number_input("Rainfall (mm)", min_value=0.0, max_value=4000.0, value=900.0, step=10.0)
        temperature_c = col3.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=28.0, step=1.0)

        col4, col5, col6 = st.columns(3)
        humidity_pct = col4.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=68.0, step=1.0)
        nitrogen = col5.number_input("Nitrogen", min_value=0.0, max_value=300.0, value=140.0, step=1.0)
        phosphorus = col6.number_input("Phosphorus", min_value=0.0, max_value=200.0, value=38.0, step=1.0)

        col7, col8, col9 = st.columns(3)
        potassium = col7.number_input("Potassium", min_value=0.0, max_value=400.0, value=245.0, step=1.0)
        season = col8.selectbox("Season", ["Kharif", "Rabi", "Perennial", "Summer", "Winter", "Monsoon", "Autumn"], index=0)
        water_availability = col9.selectbox("Water availability", ["Low", "Medium", "High"], index=1)

        submitted = st.form_submit_button("Check suitability")

    if submitted:
        selected_location = district if district != "Any" else None

        result = predict_crop_suitability(
            training_df,
            crop_name,
            soil_ph=soil_ph,
            rainfall_mm=rainfall_mm,
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            season=season,
            water_availability=water_availability,
        )

        threshold_review = evaluate_crop_thresholds(
            crop_name,
            {
                "soil_ph": soil_ph,
                "rainfall_mm": rainfall_mm,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "season": season,
                "water_availability": water_availability,
            },
        )

        try:
            yield_forecast = predict_yield_for_crop(
                training_df,
                crop_name,
                soil_ph=soil_ph,
                rainfall_mm=rainfall_mm,
                temperature_c=temperature_c,
                humidity_pct=humidity_pct,
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                season=season,
                water_availability=water_availability,
                district=selected_location,
            )
        except Exception as exc:
            yield_forecast = {"status": "fallback_only", "predicted_yield_ton_per_hectare": None, "notes": [f"Yield forecast unavailable: {exc}"]}

        try:
            recommended_crops = recommend_best_crops_for_region(
                training_df,
                district=selected_location,
                climate=climate if climate != "Any" else None,
                top_n=3,
            )
        except Exception:
            recommended_crops = []

        if district != "Any":
            selected_crop_match = next(
                (item for item in recommended_crops if str(item.get("crop", "")).strip().lower() == str(crop_name).strip().lower()),
                None,
            )
            regional_score = float(selected_crop_match.get("score", 0) or 0) if selected_crop_match else 0.0
            if selected_crop_match is None or regional_score < 0.5:
                result["confidence"] = min(float(result.get("confidence", 0.0)), 0.65)
                if result["confidence"] >= 0.5:
                    result["suitability_label"] = "Moderately suitable"
                    result["suitable"] = True
                else:
                    result["suitability_label"] = "Low suitability"
                    result["suitable"] = False

        badge_color = "#2e9d64" if result["suitable"] else "#c45b3d"
        st.markdown(
            f"<div class='glass-card' style='margin-top: 1rem; border-left: 6px solid {badge_color};'><div style='font-size: 0.8rem; color:#5f6f72;'>Prediction</div><div style='font-size: 1.8rem; font-weight: 800; color:#1d3557;'>{result['crop']} — {result['suitability_label']}</div><div style='font-size: 1rem; color:#3a4d59;'>Selected crop confidence: {result['confidence'] * 100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )

        for reason in result["reasons"]:
            st.write(f"- {reason}")

        yield_value = yield_forecast.get("predicted_yield_ton_per_hectare")
        if yield_value is not None:
            st.info(f"Estimated yield: {yield_value:.2f} tons/hectare under this profile.")

        st.subheader("Regional crop suitability")
        if recommended_crops:
            region_df = pd.DataFrame(recommended_crops)[["crop", "score", "avg_yield_ton_per_hectare", "avg_rainfall_mm", "avg_soil_ph"]]
            if "crop" in region_df.columns:
                selected_crop_lower = str(crop_name).strip().lower()
                region_df["selected_crop"] = region_df["crop"].astype(str).str.lower() == selected_crop_lower
                region_df = region_df.sort_values(["selected_crop", "score"], ascending=[False, False]).reset_index(drop=True)
                region_df = region_df.drop(columns=["selected_crop"], errors="ignore")
            st.dataframe(region_df, width="stretch")

            top_crop = recommended_crops[0]["crop"] if recommended_crops else crop_name
            selected_crop_rank = next((idx + 1 for idx, item in enumerate(recommended_crops) if str(item.get("crop", "")).strip().lower() == str(crop_name).strip().lower()), None)
            if selected_crop_rank:
                st.success(f"Best crop in this region: {top_crop}. Your selected crop '{crop_name}' ranks #{selected_crop_rank} among the region's strongest matches.")
            else:
                st.success(f"Best crop in this region: {top_crop}. The current selection is not in the top regional matches, but it may still be suitable depending on local conditions.")
        else:
            st.info("No regional crop ranking was available for this district/selection.")

        st.subheader("Crop-specific threshold review")
        threshold_rows = []
        for check in threshold_review["checks"]:
            cleaned = dict(check)
            for key in ["value", "target"]:
                if cleaned.get(key) is None:
                    cleaned[key] = ""
                elif isinstance(cleaned[key], (list, tuple, set)):
                    cleaned[key] = ", ".join(str(item) for item in cleaned[key])
            threshold_rows.append(cleaned)
        threshold_df = pd.DataFrame(threshold_rows)
        if not threshold_df.empty:
            st.dataframe(threshold_df, width="stretch")
        if threshold_review["issues"]:
            for issue in threshold_review["issues"]:
                st.warning(issue)

        st.subheader("District and climate recommendation")
        if recommended_crops:
            st.dataframe(pd.DataFrame(recommended_crops), width="stretch")
        else:
            st.info("No strong regional crop recommendation was available from the uploaded data.")

        llm_explanation = generate_agronomy_explanation(
            crop_name,
            {
                "soil_ph": soil_ph,
                "rainfall_mm": rainfall_mm,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "season": season,
                "water_availability": water_availability,
            },
            district=district if district != "Any" else None,
            climate=climate if climate != "Any" else None,
            yield_estimate=yield_value,
            suitability=result.get("suitability_label"),
            confidence=result.get("confidence"),
        )

        try:
            pdf_bytes = generate_crop_report_pdf(
                crop_name,
                result,
                yield_forecast,
                threshold_review,
                recommended_crops,
                llm_explanation,
                selected_region=district if district != "Any" else None,
            )
            st.download_button(
                label="Download report as PDF",
                data=pdf_bytes,
                file_name=f"{str(crop_name).strip().lower().replace(' ', '_')}_crop_report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.warning(f"PDF export is unavailable right now: {exc}")

        st.subheader("AI agronomy explanation")
        st.write(llm_explanation.get("summary", "No summary available."))
        with st.expander("Risks"):
            for risk in llm_explanation.get("risks", []):
                st.write(f"- {risk}")
        with st.expander("Recommendations"):
            for recommendation in llm_explanation.get("recommendations", []):
                st.write(f"- {recommendation}")
        with st.expander("Next actions"):
            for action in llm_explanation.get("next_actions", []):
                st.write(f"- {action}")


def render_dashboard() -> None:
    load_css()
    st.sidebar.markdown('<div class="sidebar-content"><h3 style="color:#ffffff; margin-bottom:0.2rem;">Welcome</h3><p style="color:#dfeef6; margin:0 0 1rem 0;">%s</p></div>' % st.session_state['username'], unsafe_allow_html=True)
    is_admin = str(st.session_state['username']).lower() == 'admin'
    if is_admin:
        st.sidebar.markdown('<div style="color:#dfeef6; font-weight:700; margin-top:0.5rem;">Role: Admin</div>', unsafe_allow_html=True)
    st.sidebar.button("Logout", on_click=lambda: st.session_state.pop("username", None))

    recent_sessions = get_recent_sessions(st.session_state["username"], limit=5)
    if recent_sessions:
        st.sidebar.markdown('<div style="color:#dfeef6; font-weight:700; margin-top:1rem; margin-bottom:0.5rem;">Recent sessions</div>', unsafe_allow_html=True)
        for item in recent_sessions:
            file_name = item["file_name"] or "Unnamed session"
            summary = item["summary"] or "Uploaded dataset"
            timestamp = item["created_at"] or "just now"
            st.sidebar.markdown(
                f"<div style='color:#eaf5ff; background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.6rem 0.7rem; margin-bottom: 0.45rem;'><div style='font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{file_name}</div><div style='font-size:0.72rem; color:#bfd7eb; margin-top:0.2rem;'>{summary}</div><div style='font-size:0.68rem; color:#a9c2d8; margin-top:0.2rem;'>{timestamp}</div></div>",
                unsafe_allow_html=True,
            )

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
    st.markdown(
        """
        <style>
        .stButton > button {
            width: 100%;
            min-height: 48px;
            border-radius: 14px;
            font-weight: 700;
        }
        .control-row {
            margin-bottom: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tabs = ["Overview", "Crop Suitability Checker"]
    if is_admin:
        tabs.append("Admin Pipeline View")
    if is_admin:
        overview_tab, crop_tab, admin_tab = st.tabs(tabs)
    else:
        overview_tab, crop_tab = st.tabs(tabs)

    with overview_tab:
        uploaded_files = st.file_uploader("Upload one or more agricultural datasets", type=["csv", "xlsx", "xls", "json"], accept_multiple_files=True)

        if uploaded_files:
            uploaded_key = tuple(getattr(f, "name", "") for f in uploaded_files)
            if st.session_state.get("uploaded_key") != uploaded_key:
                st.session_state.pop("cleaned_dataset", None)
                st.session_state["uploaded_key"] = uploaded_key

        if st.button("Run AgriAgent pipeline", use_container_width=True):
            try:
                raw_df = st.session_state.get("cleaned_dataset")
                if raw_df is None:
                    if uploaded_files:
                        raw_df = clean_uploaded_data(uploaded_files)
                        st.session_state["cleaned_dataset"] = raw_df
                    else:
                        raw_df = read_default_dataset_folder()
                        st.session_state["cleaned_dataset"] = raw_df
                else:
                    raw_df = raw_df.copy()

                # Run the full pipeline, including automatic cleaning when needed.
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

                dashboard_pdf = generate_dashboard_report_pdf(report)
                st.download_button(
                    label="Download dashboard PDF",
                    data=dashboard_pdf,
                    file_name="agriagent_dashboard_report.pdf",
                    mime="application/pdf",
                )

            except Exception as exc:  # pragma: no cover
                st.error(f"The pipeline failed: {exc}")

        if not uploaded_files:
            st.info("Using the default dataset from the data folder. Upload a new file to override it.")
            default_df = read_default_dataset_folder()
            st.dataframe(default_df.head(10), width="stretch")

    with crop_tab:
        training_df = read_uploaded_files(uploaded_files) if "uploaded_files" in locals() else read_default_dataset_folder()
        render_crop_suitability_checker(training_df)

    if is_admin:
        with admin_tab:
            st.subheader("Admin pipeline audit")
            st.caption("Schema matching, ETL transformation logic, and validation as seen by the admin role.")
            admin_dataset = read_uploaded_files(uploaded_files) if "uploaded_files" in locals() else read_default_dataset_folder()
            admin_report = run_agentic_pipeline(admin_dataset)

            st.subheader("Schema matching details")
            st.dataframe(pd.DataFrame(admin_report["schema_mapping"]), width="stretch")

            st.subheader("ETL + validation process")
            st.write("- Bronze data preserved from raw uploads")
            for step in admin_report["workflow"]["etl_steps"]:
                st.write(f"- {step}")
            st.json(admin_report["validation"])

            st.subheader("Profile report")
            st.dataframe(pd.DataFrame(admin_report["profile_report"]), width="stretch")

            st.subheader("Workflow detection")
            st.json(admin_report["workflow"])


init_db()
ensure_default_user()

if "username" not in st.session_state or not st.session_state["username"]:
    render_auth_page()
else:
    render_dashboard()

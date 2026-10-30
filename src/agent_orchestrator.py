from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .analytics import generate_agri_insights, generate_agriculture_summary, generate_chart_collection
from .data_pipeline import (
    apply_schema_mapping,
    build_dataset_summary,
    build_gold_dataset,
    clean_dataset,
    detect_agriculture_dataset,
    generate_schema_mapping,
    profile_dataset,
    validate_dataset,
)


class AgenticDataOrchestrator:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df.copy()

    def run_pipeline(self) -> Dict[str, Any]:
        bronze_df = self.raw_df.copy()
        profile_report = profile_dataset(bronze_df)
        domain_context = detect_agriculture_dataset(bronze_df)

        mapping = generate_schema_mapping(bronze_df)
        mapped_df = apply_schema_mapping(bronze_df, mapping)

        silver_df = clean_dataset(mapped_df)
        gold_df = build_gold_dataset(silver_df)

        validation_report = validate_dataset(gold_df)
        summary = build_dataset_summary(gold_df)
        analytics_summary = generate_agriculture_summary(gold_df)
        insights = generate_agri_insights(gold_df)
        charts = generate_chart_collection(gold_df)

        return {
            "bronze": bronze_df,
            "silver": silver_df,
            "gold": gold_df,
            "schema_mapping": mapping,
            "profile_report": profile_report,
            "validation": validation_report,
            "summary": summary,
            "analytics_summary": analytics_summary,
            "insights": insights,
            "charts": charts,
            "workflow": {
                "domain_detected": domain_context["domain"],
                "matched_columns": domain_context["matched_columns"],
                "etl_steps": [
                    "Data profiling: identify soil, crop, weather, yield and missing-value indicators",
                    "LLM schema agent: map columns to agriculture ontology and canonical names",
                    "Bronze layer: preserve raw agronomic data for traceability",
                    "ETL transformation: normalize types, fill missing values, standardize units",
                    "Validation agent: run rule checks, duplicates, range validation, and quality gates",
                    "Silver layer: cleaned agriculture records ready for analytics",
                    "Feature agent: create soil, weather and yield features",
                    "Gold layer: analytics-ready dataset for crop intelligence and ML",
                    "Text-to-analytics and dashboard: convert results into decision-ready insights",
                ],
            },
        }


def run_agentic_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    orchestrator = AgenticDataOrchestrator(df)
    return orchestrator.run_pipeline()

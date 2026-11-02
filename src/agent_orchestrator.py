from __future__ import annotations

import io
import re
from pathlib import Path
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
    load_dataset,
    profile_dataset,
    validate_dataset,
)


class DataCleaningAgent:
    """Clean messy uploaded datasets before passing them to the preparation pipeline."""

    KNOWN_NUMERIC_COLUMNS = {
        "soil_ph",
        "rainfall_mm",
        "temperature_c",
        "humidity_pct",
        "nitrogen",
        "phosphorus",
        "potassium",
        "yield_ton_per_hectare",
        "field_area_hectare",
        "yield",
    }

    COLUMN_ALIASES = {
        "crop": {"crop", "crop_name", "crop_type", "crop_kind", "plant"},
        "soil_ph": {"soil_ph", "soilph", "ph", "ph_value", "soil_acidity"},
        "rainfall_mm": {"rainfall", "rainfall_mm", "rain_mm", "precipitation", "annual_rainfall"},
        "temperature_c": {"temperature", "temperature_c", "temp_c", "avg_temperature", "temp"},
        "humidity_pct": {"humidity", "humidity_pct", "relative_humidity", "rh"},
        "nitrogen": {"nitrogen", "n_level", "n_content", "n"},
        "phosphorus": {"phosphorus", "p_level", "p_content", "p"},
        "potassium": {"potassium", "k_level", "k_content", "k"},
        "yield_ton_per_hectare": {"yield", "yield_ton_per_hectare", "production", "output"},
        "district": {"district", "district_name", "location", "area", "region", "region_name", "state", "state_name", "province", "city", "block", "taluka", "village", "location_name"},
        "season": {"season", "crop_season", "growing_season", "sowing_season"},
        "water_availability": {"water_availability", "water_source", "irrigation", "water"},
        "soil_type": {"soil_type", "soil_class", "soil_texture"},
        "field_area_hectare": {"field_area", "land_area", "area_hectare", "hectares"},
    }

    NULL_MARKERS = {
        "", "nan", "n/a", "na", "null", "none", "nil", "unknown", "undefined", "missing", "not available"
    }

    @staticmethod
    def _normalize_column_name(column: Any) -> str:
        text = str(column).strip()
        text = text.replace("%", " pct ")
        text = text.replace("°", " ")
        text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text.lower()

    @classmethod
    def _canonicalize_columns(cls, columns: list[str]) -> list[str]:
        canonical: list[str] = []
        for col in columns:
            normalized = cls._normalize_column_name(col)
            mapped = normalized
            for candidate, aliases in cls.COLUMN_ALIASES.items():
                if normalized in aliases or normalized == candidate:
                    mapped = candidate
                    break
            canonical.append(mapped)
        return canonical

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in DataCleaningAgent.NULL_MARKERS:
            return None
        return text

    @staticmethod
    def _safe_numeric(value: Any) -> Any:
        cleaned = DataCleaningAgent._normalize_value(value)
        if cleaned is None:
            return None
        if isinstance(cleaned, (int, float)):
            return float(cleaned)
        text = str(cleaned).strip()
        text = text.replace("%", "").replace("₹", "").replace("$", "").replace("(", "").replace(")", "")
        text = text.replace(",", "")
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    @classmethod
    def _clean_string_value(cls, value: Any) -> Any:
        cleaned = cls._normalize_value(value)
        if cleaned is None:
            return None
        text = str(cleaned).strip()
        if not text:
            return None
        text = re.sub(r"\s+", " ", text)
        text = text.replace("_", " ")
        return text.title() if text.isalpha() else text

    @classmethod
    def _clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        cleaned = df.copy()
        cleaned.columns = cls._canonicalize_columns(list(cleaned.columns))
        cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()].copy()
        cleaned = cleaned.dropna(how="all").reset_index(drop=True)

        for column in cleaned.columns:
            series = cleaned[column]
            cleaned[column] = series.map(cls._normalize_value)

            non_null = cleaned[column].dropna()
            if non_null.empty:
                cleaned[column] = pd.Series([None] * len(cleaned), index=cleaned.index)
                continue

            numeric_ratio = 0.0
            if len(non_null) > 0:
                numeric_ratio = float(non_null.map(cls._safe_numeric).notna().mean())

            if column in cls.KNOWN_NUMERIC_COLUMNS or numeric_ratio >= 0.7:
                numeric_series = cleaned[column].map(cls._safe_numeric)
                cleaned[column] = pd.to_numeric(numeric_series, errors="coerce")
                median = cleaned[column].median()
                if pd.notna(median):
                    cleaned[column] = cleaned[column].fillna(median)
                continue

            cleaned[column] = cleaned[column].map(cls._clean_string_value)
            valid = cleaned[column].dropna()
            if not valid.empty:
                mode = valid.mode(dropna=True)
                if not mode.empty:
                    cleaned[column] = cleaned[column].fillna(mode.iloc[0])
                else:
                    cleaned[column] = cleaned[column].fillna("Unknown")

        cleaned = cleaned.replace({"nan": None, "None": None, "null": None, "NaN": None, "NULL": None})
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        for col in cleaned.columns:
            if cleaned[col].map(lambda x: isinstance(x, str) and x.strip() == "").any():
                cleaned[col] = cleaned[col].replace(r"^\s*$", None, regex=True)

        return cleaned

    @classmethod
    def _coerce_dataframe(cls, item: Any) -> pd.DataFrame:
        if item is None:
            return pd.DataFrame()
        if isinstance(item, pd.DataFrame):
            return cls._clean_dataframe(item)
        if isinstance(item, (str, bytes, Path)):
            return cls._clean_dataframe(load_dataset(item))
        if hasattr(item, "read") or hasattr(item, "getvalue"):
            return cls._clean_dataframe(DatasetPreparationAgent._read_uploaded_file(item))
        if isinstance(item, dict):
            return cls._clean_dataframe(pd.DataFrame([item]))
        if isinstance(item, (list, tuple)):
            return cls._clean_dataframe(pd.DataFrame(item))
        return cls._clean_dataframe(pd.DataFrame(item))

    @classmethod
    def clean(cls, raw_datasets: Any) -> list[pd.DataFrame] | pd.DataFrame:
        if raw_datasets is None:
            return []

        if isinstance(raw_datasets, (pd.DataFrame, str, bytes, Path)) or hasattr(raw_datasets, "read") or hasattr(raw_datasets, "getvalue"):
            coerced = cls._coerce_dataframe(raw_datasets)
            return coerced if not coerced.empty else pd.DataFrame()

        if isinstance(raw_datasets, (list, tuple)):
            cleaned_frames = []
            for item in raw_datasets:
                if item is None:
                    continue
                frame = cls._coerce_dataframe(item)
                if not frame.empty:
                    cleaned_frames.append(frame)
            return cleaned_frames

        coerced = cls._coerce_dataframe(raw_datasets)
        return coerced if not coerced.empty else pd.DataFrame()

    @staticmethod
    def prepare(raw_datasets: Any) -> Dict[str, Any]:
        cleaned = DataCleaningAgent.clean(raw_datasets)
        if isinstance(cleaned, list):
            return DatasetPreparationAgent(cleaned).prepare()
        if not cleaned.empty:
            return DatasetPreparationAgent([cleaned]).prepare()
        raise ValueError("No valid datasets were provided for cleaning and preparation.")


class DatasetPreparationAgent:
    """Clean and normalize one or many uploaded datasets for downstream analytics."""

    def __init__(self, raw_datasets: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...] | str | None):
        self.raw_datasets = self._normalize_inputs(raw_datasets)

    @staticmethod
    def _read_uploaded_file(file_obj: Any) -> pd.DataFrame:
        if hasattr(file_obj, "getvalue"):
            payload = file_obj.getvalue()
        elif hasattr(file_obj, "read"):
            payload = file_obj.read()
        else:
            raise TypeError("Unsupported uploaded dataset object.")

        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("Uploaded dataset payload must be bytes or text content.")

        suffix = Path(str(getattr(file_obj, "name", "uploaded_dataset"))).suffix.lower()
        raw_bytes = io.BytesIO(payload)

        if suffix == ".csv":
            return pd.read_csv(raw_bytes)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(raw_bytes)
        if suffix == ".json":
            return pd.read_json(raw_bytes)

        # Fallback for generic upload objects: try CSV first, then Excel, then JSON.
        for reader, extension in [
            (pd.read_csv, ".csv"),
            (pd.read_excel, ".xlsx"),
            (pd.read_json, ".json"),
        ]:
            try:
                raw_bytes.seek(0)
                return reader(raw_bytes)
            except Exception:
                raw_bytes.seek(0)
        raise ValueError(f"Unsupported uploaded dataset format for file name: {getattr(file_obj, 'name', 'uploaded_dataset')}")

    @classmethod
    def _coerce_dataframe(cls, item: Any) -> pd.DataFrame:
        if item is None:
            return pd.DataFrame()
        if isinstance(item, pd.DataFrame):
            return item.copy()
        if isinstance(item, (str, bytes)):
            return load_dataset(item)
        if isinstance(item, (Path,)):
            return load_dataset(item)
        if hasattr(item, "read") or hasattr(item, "getvalue"):
            return cls._read_uploaded_file(item)
        if isinstance(item, dict):
            return pd.DataFrame([item])
        if isinstance(item, (list, tuple)):
            return pd.DataFrame(item)
        return pd.DataFrame(item)

    @classmethod
    def _normalize_inputs(cls, raw_datasets: Any) -> list[pd.DataFrame]:
        if raw_datasets is None:
            return []

        if isinstance(raw_datasets, (pd.DataFrame, str, bytes, Path)) or hasattr(raw_datasets, "read") or hasattr(raw_datasets, "getvalue"):
            coerced = cls._coerce_dataframe(raw_datasets)
            return [coerced] if not coerced.empty else []

        if isinstance(raw_datasets, (list, tuple)):
            frames: list[pd.DataFrame] = []
            for item in raw_datasets:
                if item is None:
                    continue
                frame = cls._coerce_dataframe(item)
                if frame is not None and not frame.empty:
                    frames.append(frame)
            return frames

        coerced = cls._coerce_dataframe(raw_datasets)
        return [coerced] if not coerced.empty else []

    def prepare(self) -> Dict[str, Any]:
        prepared_datasets: list[Dict[str, Any]] = []
        combined_df = pd.DataFrame()

        for index, frame in enumerate(self.raw_datasets):
            if frame is None or frame.empty:
                continue
            bronze_df = frame.copy()
            profile_report = profile_dataset(bronze_df)
            domain_context = detect_agriculture_dataset(bronze_df)

            mapping = generate_schema_mapping(bronze_df)
            mapped_df = apply_schema_mapping(bronze_df, mapping)
            silver_df = clean_dataset(mapped_df)
            gold_df = build_gold_dataset(silver_df)
            validation_report = validate_dataset(gold_df)
            summary = build_dataset_summary(gold_df)

            prepared_datasets.append(
                {
                    "dataset_index": index,
                    "bronze": bronze_df,
                    "silver": silver_df,
                    "gold": gold_df,
                    "schema_mapping": mapping,
                    "profile_report": profile_report,
                    "validation": validation_report,
                    "summary": summary,
                    "domain_detected": domain_context["domain"],
                    "matched_columns": domain_context["matched_columns"],
                }
            )

            if combined_df.empty:
                combined_df = gold_df.copy()
            else:
                combined_df = pd.concat([combined_df, gold_df], ignore_index=True, sort=False)

        if not prepared_datasets:
            raise ValueError("No valid datasets were provided for cleaning and preparation.")

        dataset_summary = build_dataset_summary(combined_df)
        workflow = {
            "dataset_count": len(prepared_datasets),
            "domain_detected": prepared_datasets[0]["domain_detected"],
            "matched_columns": prepared_datasets[0]["matched_columns"],
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
        }

        return {
            "datasets": prepared_datasets,
            "prepared_dataset": combined_df,
            "summary": dataset_summary,
            "workflow": workflow,
            "validation": prepared_datasets[-1]["validation"],
        }


class AgenticDataOrchestrator:
    def __init__(self, raw_df: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...] | str | None):
        self.raw_df = self._coerce_dataframe(raw_df)

    @staticmethod
    def _coerce_dataframe(raw_df: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...] | str | None) -> pd.DataFrame:
        if raw_df is None:
            return pd.DataFrame()

        cleaned = DataCleaningAgent.clean(raw_df)
        if isinstance(cleaned, list):
            if not cleaned:
                return pd.DataFrame()
            merged = pd.concat(cleaned, ignore_index=True, sort=False)
            return merged.reset_index(drop=True)
        if cleaned.empty:
            return pd.DataFrame()
        return cleaned.reset_index(drop=True)

    def run_pipeline(self) -> Dict[str, Any]:
        if self.raw_df.empty:
            raise ValueError("No dataset provided to the AgriAgent pipeline.")

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


def prepare_uploaded_datasets(raw_datasets: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...] | str | None) -> Dict[str, Any]:
    return DataCleaningAgent.prepare(raw_datasets)


def run_agentic_pipeline(df: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...] | str | None) -> Dict[str, Any]:
    cleaned = DataCleaningAgent.clean(df)
    if isinstance(cleaned, list):
        if not cleaned:
            raise ValueError("No dataset provided to the AgriAgent pipeline.")
        merged = pd.concat(cleaned, ignore_index=True, sort=False)
        df = merged.reset_index(drop=True)
    else:
        df = cleaned.reset_index(drop=True) if not cleaned.empty else pd.DataFrame()
    orchestrator = AgenticDataOrchestrator(df)
    return orchestrator.run_pipeline()


__all__ = [
    "DataCleaningAgent",
    "DatasetPreparationAgent",
    "AgenticDataOrchestrator",
    "prepare_uploaded_datasets",
    "run_agentic_pipeline",
]

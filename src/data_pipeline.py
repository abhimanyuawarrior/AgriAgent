import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .llm import call_gemini_for_mapping

CANONICAL_FIELDS: Dict[str, List[str]] = {
    "crop": ["crop", "crop_name", "crop_type", "crop_kind", "plant", "crop_variety", "crop_species", "commodity"],
    "soil_ph": ["soil_ph", "soilph", "soil_p_h", "ph", "ph_value", "soil_acidity", "soil_alkalinity"],
    "rainfall_mm": ["rainfall", "rainfall_mm", "rain_mm", "precipitation", "annual_rainfall", "rain", "rainfall_amount", "mm_rainfall"],
    "temperature_c": ["temperature", "temperature_c", "temp_c", "avg_temperature", "temp", "air_temp", "mean_temp"],
    "humidity_pct": ["humidity", "humidity_pct", "relative_humidity", "rh", "moisture", "air_humidity"],
    "nitrogen": ["nitrogen", "n_level", "n_content", "n", "nitrogen_content", "n_value"],
    "phosphorus": ["phosphorus", "p_level", "p_content", "p", "phosphorous", "phosphorus_value"],
    "potassium": ["potassium", "k_level", "k_content", "k", "potash", "potassium_value"],
    "yield_ton_per_hectare": ["yield", "yield_ton_per_hectare", "production", "output", "yield_t ha", "yield_per_hectare", "harvest", "production_ton_hectare"],
    "district": ["district", "location", "area", "region", "state", "block", "taluka", "village"],
    "season": ["season", "crop_season", "growing_season", "sowing_season", "month"],
    "water_availability": ["water_availability", "water_source", "irrigation", "water", "water_supply", "canal"],
    "soil_type": ["soil_type", "soil_class", "soil_texture", "soilcategory"],
    "field_area_hectare": ["field_area", "land_area", "area_hectare", "hectares", "farm_area"],
    "fertilizer_used": ["fertilizer", "fertilizer_type", "nutrient_input", "fertilizer_used"],
    "crop_health": ["crop_health", "plant_health", "health_status", "vegetation_index", "ndvi"],
}

NUMERIC_FIELDS = {
    "soil_ph",
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "nitrogen",
    "phosphorus",
    "potassium",
    "yield_ton_per_hectare",
}

TEXT_FIELDS = {"crop", "district", "season", "water_availability"}


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def alias_similarity(source: str, target: str) -> float:
    norm_source = normalize_name(source)
    norm_target = normalize_name(target)
    if not norm_source or not norm_target:
        return 0.0
    if norm_source == norm_target:
        return 1.0
    if norm_target in norm_source.split() or norm_source in norm_target.split():
        return 0.8
    return SequenceMatcher(None, norm_source, norm_target).ratio()


def score_field_match(column_name: str, canonical_field: str) -> float:
    aliases = CANONICAL_FIELDS.get(canonical_field, [])
    best = 0.0
    for alias in [canonical_field, *aliases]:
        score = alias_similarity(column_name, alias)
        if score > best:
            best = score
    return best


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def profile_dataset(df: pd.DataFrame) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round(float(series.isna().mean() * 100), 2)
        sample_values = [str(value) for value in series.dropna().head(5).tolist()]

        non_null_values = series.dropna()
        mixed_types = False
        if len(non_null_values) > 0:
            value_types = {type(value).__name__ for value in non_null_values.head(50).tolist()}
            mixed_types = len(value_types) > 1

        quality_flags = []
        if missing_count > 0:
            quality_flags.append("missing_values")
        if mixed_types:
            quality_flags.append("mixed_types")
        if series.duplicated().mean() > 0.05:
            quality_flags.append("duplicate_values")
        if col.lower().endswith(("_ph", "_mm", "_c", "_pct")) or col.lower() in {"soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus", "potassium", "yield_ton_per_hectare"}:
            numeric_like = pd.to_numeric(series, errors="coerce")
            if numeric_like.notna().sum() > 0 and (numeric_like.isna().sum() / max(len(series), 1)) > 0.1:
                quality_flags.append("non_numeric_values")

        profiles.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": sample_values[:5],
                "quality_flags": quality_flags,
                "messy_data": bool(quality_flags),
            }
        )
    return profiles


def detect_agriculture_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    matched = []
    for column in df.columns:
        best_target, best_score = None, 0.0
        for canonical_name in CANONICAL_FIELDS:
            score = score_field_match(str(column), canonical_name)
            if score > best_score:
                best_target = canonical_name
                best_score = score
        if best_target and best_score >= 0.35:
            matched.append({"column": str(column), "mapped_to": best_target, "score": round(best_score, 3)})

    agriculture_score = len(matched)
    inferred_domain = "agriculture" if agriculture_score >= 3 else "general_tabular"
    return {
        "domain": inferred_domain,
        "matched_columns": matched,
        "match_count": agriculture_score,
    }


def generate_schema_mapping(df: pd.DataFrame) -> List[Dict[str, Any]]:
    llm_mapping = None
    if os.getenv("GEMINI_API_KEY"):
        llm_mapping = call_gemini_for_mapping(list(df.columns))

    if llm_mapping:
        normalized = []
        seen = set()
        for item in llm_mapping:
            source = str(item.get("source_column", "")).strip()
            target = str(item.get("mapped_to", "")).strip()
            if not source or not target:
                continue
            if source not in df.columns:
                continue
            if source in seen:
                continue
            seen.add(source)
            normalized.append({
                "source_column": source,
                "mapped_to": target,
                "score": float(item.get("confidence", item.get("score", 0.8))) if isinstance(item.get("confidence", item.get("score", 0.8)), (int, float)) else 0.8,
            })
        if normalized:
            return normalized

    mapping: List[Dict[str, Any]] = []
    for column in df.columns:
        best_target, best_score = None, 0.0
        for canonical_name in CANONICAL_FIELDS:
            score = score_field_match(str(column), canonical_name)
            if score > best_score:
                best_target = canonical_name
                best_score = score
        if best_target is not None and best_score >= 0.35:
            mapping.append({
                "source_column": str(column),
                "mapped_to": best_target,
                "score": round(best_score, 3),
            })
    return mapping


def apply_schema_mapping(df: pd.DataFrame, mapping: List[Dict[str, Any]]) -> pd.DataFrame:
    result = df.copy()
    used_targets: set[str] = set()
    for match in mapping:
        target = match["mapped_to"]
        if target in used_targets:
            continue
        source = match["source_column"]
        if source == target:
            continue
        if target in result.columns:
            result = result.rename(columns={source: f"{target}_source"})
        else:
            result = result.rename(columns={source: target})
        used_targets.add(target)
    return result


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in cleaned.columns:
        series = cleaned[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        cleaned[column] = series.map(lambda value: str(value).strip() if pd.notna(value) else None)

    for column in cleaned.columns:
        if column in NUMERIC_FIELDS:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            median = cleaned[column].median()
            if pd.notna(median):
                cleaned[column] = cleaned[column].fillna(median)
        else:
            cleaned[column] = cleaned[column].fillna("Unknown")

    cleaned = cleaned.drop_duplicates()
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def build_gold_dataset(df: pd.DataFrame) -> pd.DataFrame:
    gold = df.copy()

    if "soil_ph" in gold.columns:
        gold["soil_quality_score"] = gold["soil_ph"].apply(
            lambda ph: max(0, min(100, 100 - abs(float(ph) - 6.5) * 15))
        )

    if "rainfall_mm" in gold.columns:
        gold["rainfall_suitability"] = gold["rainfall_mm"].apply(
            lambda value: max(0, min(100, (float(value) / 1200) * 100))
        )

    if "temperature_c" in gold.columns:
        gold["temperature_suitability"] = gold["temperature_c"].apply(
            lambda value: max(0, min(100, 100 - abs(float(value) - 28) * 4))
        )

    if "yield_ton_per_hectare" in gold.columns:
        gold["yield_status"] = gold["yield_ton_per_hectare"].apply(
            lambda value: "High" if float(value) >= 4.0 else "Moderate" if float(value) >= 2.5 else "Low"
        )

    if all(col in gold.columns for col in ["soil_ph", "rainfall_mm", "temperature_c"]):
        gold["agri_health_index"] = (
            gold["soil_quality_score"] * 0.4
            + gold["rainfall_suitability"] * 0.4
            + gold["temperature_suitability"] * 0.2
        )

    return gold


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    issues: List[str] = []
    checks = []
    missing_value_columns = []

    for column in ["soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "yield_ton_per_hectare"]:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            missing_ratio = float(series.isna().mean() * 100)
            checks.append({
                "column": column,
                "missing_pct": round(missing_ratio, 2),
                "min": round(float(series.min()), 2) if series.notna().any() else None,
                "max": round(float(series.max()), 2) if series.notna().any() else None,
            })
            if missing_ratio > 0:
                missing_value_columns.append(column)
            if missing_ratio > 25:
                issues.append(f"{column} has more than 25% missing values.")

    if "soil_ph" in df.columns:
        soil_values = pd.to_numeric(df["soil_ph"], errors="coerce")
        if (soil_values < 3).any() or (soil_values > 10).any():
            issues.append("soil_ph contains values outside the expected 3-10 range.")

    if "rainfall_mm" in df.columns:
        rainfall_values = pd.to_numeric(df["rainfall_mm"], errors="coerce")
        if (rainfall_values < 0).any():
            issues.append("rainfall_mm contains negative values.")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        issues.append(f"Detected {duplicate_rows} duplicate rows.")

    validation_status = "passed" if not issues else "warning"
    return {
        "status": validation_status,
        "duplicate_rows": duplicate_rows,
        "missing_value_columns": missing_value_columns,
        "checks": checks,
        "issues": issues,
    }


def build_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "crops": sorted(df["crop"].dropna().astype(str).unique().tolist()) if "crop" in df.columns else [],
        "avg_soil_ph": round(float(df["soil_ph"].mean()), 2) if "soil_ph" in df.columns else None,
        "avg_rainfall_mm": round(float(df["rainfall_mm"].mean()), 2) if "rainfall_mm" in df.columns else None,
        "avg_yield_ton_per_hectare": round(float(df["yield_ton_per_hectare"].mean()), 2) if "yield_ton_per_hectare" in df.columns else None,
    }
    return summary


def export_json_serializable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): export_json_serializable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [export_json_serializable(item) for item in payload]
    if isinstance(payload, tuple):
        return [export_json_serializable(item) for item in payload]
    if pd.isna(payload):
        return None
    return payload

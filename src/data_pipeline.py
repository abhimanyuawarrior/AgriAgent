import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def train_crop_suitability_model(df: pd.DataFrame) -> Dict[str, Any]:
    if "crop" not in df.columns:
        return {"model": None, "status": "missing_crop_column", "feature_columns": []}

    working = df.copy()
    working["crop"] = working["crop"].astype(str).str.strip()
    working = working.dropna(subset=["crop"]).copy()
    if working.empty:
        return {"model": None, "status": "empty_dataset", "feature_columns": []}

    feature_columns = [
        col for col in ["soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus", "potassium", "season", "water_availability", "soil_type"]
        if col in working.columns
    ]
    if not feature_columns:
        return {"model": None, "status": "no_agri_features", "feature_columns": []}

    numeric_features = [col for col in feature_columns if col in {"soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus", "potassium"}]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )

    model.fit(working[feature_columns], working["crop"])
    return {
        "model": model,
        "status": "trained",
        "feature_columns": feature_columns,
        "class_names": list(model.named_steps["classifier"].classes_),
    }


def predict_crop_suitability(
    df: pd.DataFrame,
    crop_name: str,
    soil_ph: float | None = None,
    rainfall_mm: float | None = None,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    nitrogen: float | None = None,
    phosphorus: float | None = None,
    potassium: float | None = None,
    season: str | None = None,
    water_availability: str | None = None,
) -> Dict[str, Any]:
    if crop_name is None or str(crop_name).strip() == "":
        return {"status": "invalid_crop", "message": "Please provide a valid crop name."}

    crop_name = str(crop_name).strip()
    training_state = train_crop_suitability_model(df)
    model = training_state.get("model")

    if model is None:
        return {
            "status": "fallback_only",
            "crop": crop_name,
            "suitability_label": "Needs more field data",
            "confidence": 0.0,
            "suitable": False,
            "reasons": ["Add crop, soil, rainfall, temperature, and nutrient values to train the suitability model."],
        }

    feature_columns = training_state["feature_columns"]
    input_row: Dict[str, Any] = {}

    for feature in feature_columns:
        if feature == "soil_ph":
            input_row[feature] = soil_ph if soil_ph is not None else df[feature].median()
        elif feature == "rainfall_mm":
            input_row[feature] = rainfall_mm if rainfall_mm is not None else df[feature].median()
        elif feature == "temperature_c":
            input_row[feature] = temperature_c if temperature_c is not None else df[feature].median()
        elif feature == "humidity_pct":
            input_row[feature] = humidity_pct if humidity_pct is not None else df[feature].median()
        elif feature == "nitrogen":
            input_row[feature] = nitrogen if nitrogen is not None else df[feature].median()
        elif feature == "phosphorus":
            input_row[feature] = phosphorus if phosphorus is not None else df[feature].median()
        elif feature == "potassium":
            input_row[feature] = potassium if potassium is not None else df[feature].median()
        elif feature == "season":
            input_row[feature] = season if season is not None else df[feature].mode().iloc[0] if not df[feature].mode().empty else "Kharif"
        elif feature == "water_availability":
            input_row[feature] = water_availability if water_availability is not None else df[feature].mode().iloc[0] if not df[feature].mode().empty else "Medium"
        elif feature == "soil_type":
            input_row[feature] = df[feature].mode().iloc[0] if not df[feature].mode().empty else "Loamy"

    input_df = pd.DataFrame([input_row], columns=feature_columns)
    probability_values = model.predict_proba(input_df)[0]
    class_names = training_state["class_names"]
    prediction_index = 0
    if crop_name.lower() in {name.lower() for name in class_names}:
        prediction_index = next(i for i, name in enumerate(class_names) if name.lower() == crop_name.lower())
    confidence = float(probability_values[prediction_index]) if prediction_index < len(probability_values) else float(probability_values.max())
    if crop_name.lower() not in {name.lower() for name in class_names}:
        confidence = min(0.6, float(probability_values.max()))

    suitability = "Highly suitable" if confidence >= 0.75 else "Moderately suitable" if confidence >= 0.5 else "Low suitability"
    suitable = confidence >= 0.5
    reasons: List[str] = []

    crop_history = df[df["crop"].astype(str).str.lower() == crop_name.lower()].copy() if "crop" in df.columns else pd.DataFrame()
    if not crop_history.empty:
        for metric, value in [("soil_ph", soil_ph), ("rainfall_mm", rainfall_mm), ("temperature_c", temperature_c), ("humidity_pct", humidity_pct)]:
            if value is not None and metric in crop_history.columns:
                mean_value = float(crop_history[metric].mean())
                delta = abs(float(value) - mean_value)
                if metric == "soil_ph":
                    reasons.append(f"Soil pH {value} is {'close to' if delta <= 0.5 else 'near'} the historical range for {crop_name}.")
                else:
                    reasons.append(f"{metric.replace('_', ' ').title()} {value} aligns with the historical range used for {crop_name}.")
    if not reasons:
        reasons.append(f"The model trained on the current agriculture dataset estimates {crop_name} is {suitability.lower()} in these conditions.")

    return {
        "status": "ok",
        "crop": crop_name,
        "suitability_label": suitability,
        "confidence": round(confidence, 3),
        "suitable": suitable,
        "reasons": reasons,
        "model_class": class_names,
    }


CROP_THRESHOLD_RULES: Dict[str, Dict[str, Any]] = {
    "rice": {
        "soil_ph": (5.5, 7.5),
        "rainfall_mm": (800, 1800),
        "temperature_c": (20, 35),
        "humidity_pct": (60, 90),
        "nitrogen": (100, 180),
        "phosphorus": (20, 60),
        "potassium": (180, 300),
        "season": ["Kharif", "Monsoon"],
        "water_availability": ["High", "Medium"],
    },
    "wheat": {
        "soil_ph": (6.0, 7.5),
        "rainfall_mm": (450, 1000),
        "temperature_c": (10, 25),
        "humidity_pct": (40, 70),
        "nitrogen": (80, 150),
        "phosphorus": (20, 45),
        "potassium": (150, 260),
        "season": ["Rabi", "Winter"],
        "water_availability": ["Medium", "Low"],
    },
    "cotton": {
        "soil_ph": (5.8, 7.5),
        "rainfall_mm": (600, 1200),
        "temperature_c": (25, 35),
        "humidity_pct": (45, 70),
        "nitrogen": (90, 170),
        "phosphorus": (20, 55),
        "potassium": (150, 280),
        "season": ["Kharif", "Summer"],
        "water_availability": ["Medium", "High"],
    },
    "soybean": {
        "soil_ph": (6.0, 7.0),
        "rainfall_mm": (500, 1000),
        "temperature_c": (20, 30),
        "humidity_pct": (50, 75),
        "nitrogen": (80, 150),
        "phosphorus": (25, 50),
        "potassium": (160, 260),
        "season": ["Kharif", "Monsoon"],
        "water_availability": ["Medium", "High"],
    },
    "maize": {
        "soil_ph": (5.8, 7.2),
        "rainfall_mm": (500, 1200),
        "temperature_c": (20, 32),
        "humidity_pct": (55, 80),
        "nitrogen": (100, 170),
        "phosphorus": (25, 50),
        "potassium": (160, 280),
        "season": ["Kharif", "Summer"],
        "water_availability": ["Medium", "High"],
    },
    "sugarcane": {
        "soil_ph": (6.0, 8.0),
        "rainfall_mm": (1000, 1800),
        "temperature_c": (25, 38),
        "humidity_pct": (60, 85),
        "nitrogen": (120, 200),
        "phosphorus": (30, 60),
        "potassium": (180, 320),
        "season": ["Perennial", "Kharif"],
        "water_availability": ["High"],
    },
    "groundnut": {
        "soil_ph": (5.5, 7.0),
        "rainfall_mm": (400, 900),
        "temperature_c": (20, 32),
        "humidity_pct": (45, 75),
        "nitrogen": (80, 160),
        "phosphorus": (20, 45),
        "potassium": (120, 240),
        "season": ["Kharif", "Summer"],
        "water_availability": ["Medium", "Low"],
    },
    "grapes": {
        "soil_ph": (6.2, 7.5),
        "rainfall_mm": (500, 900),
        "temperature_c": (18, 32),
        "humidity_pct": (45, 70),
        "nitrogen": (120, 180),
        "phosphorus": (30, 55),
        "potassium": (180, 300),
        "season": ["Perennial", "Summer"],
        "water_availability": ["Medium", "High"],
    },
}


def evaluate_crop_thresholds(crop_name: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
    crop_key = str(crop_name).strip().lower()
    rules = CROP_THRESHOLD_RULES.get(crop_key)
    if not rules:
        return {
            "crop": crop_name,
            "status": "unknown_crop",
            "checks": [],
            "issues": ["No crop-specific threshold rules are available for this crop yet."],
        }

    checks: List[Dict[str, Any]] = []
    issues: List[str] = []
    for metric, bounds in rules.items():
        if metric in {"season", "water_availability"}:
            value = conditions.get(metric)
            allowed = bounds
            if value is not None and value not in allowed:
                issues.append(f"{metric.replace('_', ' ').title()} is {value}, but {crop_name.title()} prefers {', '.join(map(str, allowed))}.")
            checks.append({"metric": metric, "value": value, "target": allowed, "status": "pass" if value in allowed else "warning"})
            continue

        value = conditions.get(metric)
        if value is None:
            checks.append({"metric": metric, "value": None, "target": bounds, "status": "missing"})
            continue
        low, high = bounds
        status = "pass"
        if value < low:
            status = "low"
            issues.append(f"{metric.replace('_', ' ').title()} is below the preferred range for {crop_name.title()}.")
        elif value > high:
            status = "high"
            issues.append(f"{metric.replace('_', ' ').title()} is above the preferred range for {crop_name.title()}.")
        checks.append({"metric": metric, "value": value, "target": bounds, "status": status})

    status = "good" if not issues else "warning"
    return {"crop": crop_name, "status": status, "checks": checks, "issues": issues}


def train_yield_model(df: pd.DataFrame) -> Any | None:
    if "yield_ton_per_hectare" not in df.columns:
        return None

    working = df.copy()
    feature_columns = [
        col for col in ["soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus", "potassium", "season", "water_availability", "crop", "district"]
        if col in working.columns
    ]
    if not feature_columns:
        return None

    numeric_features = [col for col in feature_columns if col in {"soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus", "potassium"}]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
    ])
    model.fit(working[feature_columns], working["yield_ton_per_hectare"])
    return model


def predict_yield_for_crop(
    df: pd.DataFrame,
    crop_name: str,
    soil_ph: float | None = None,
    rainfall_mm: float | None = None,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    nitrogen: float | None = None,
    phosphorus: float | None = None,
    potassium: float | None = None,
    season: str | None = None,
    water_availability: str | None = None,
    district: str | None = None,
) -> Dict[str, Any]:
    if "yield_ton_per_hectare" not in df.columns:
        return {"status": "no_yield_column", "predicted_yield_ton_per_hectare": None, "notes": ["This dataset does not include a target yield field, so yield forecasting is unavailable."]}

    model = train_yield_model(df)
    if model is None:
        return {"status": "no_yield_column", "predicted_yield_ton_per_hectare": None, "notes": ["Yield forecast model could not be trained from the available dataset."]}

    feature_columns = [col for col in ["soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus", "potassium", "season", "water_availability", "crop", "district"] if col in df.columns]
    if not feature_columns:
        return {"status": "insufficient_features", "predicted_yield_ton_per_hectare": None, "notes": ["The dataset does not contain enough agronomic variables to estimate yield."]}

    input_row: Dict[str, Any] = {}
    for feature in feature_columns:
        if feature == "soil_ph":
            input_row[feature] = soil_ph if soil_ph is not None else df[feature].median()
        elif feature == "rainfall_mm":
            input_row[feature] = rainfall_mm if rainfall_mm is not None else df[feature].median()
        elif feature == "temperature_c":
            input_row[feature] = temperature_c if temperature_c is not None else df[feature].median()
        elif feature == "humidity_pct":
            input_row[feature] = humidity_pct if humidity_pct is not None else df[feature].median()
        elif feature == "nitrogen":
            input_row[feature] = nitrogen if nitrogen is not None else df[feature].median()
        elif feature == "phosphorus":
            input_row[feature] = phosphorus if phosphorus is not None else df[feature].median()
        elif feature == "potassium":
            input_row[feature] = potassium if potassium is not None else df[feature].median()
        elif feature == "season":
            input_row[feature] = season if season is not None else df[feature].mode().iloc[0] if not df[feature].mode().empty else "Kharif"
        elif feature == "water_availability":
            input_row[feature] = water_availability if water_availability is not None else df[feature].mode().iloc[0] if not df[feature].mode().empty else "Medium"
        elif feature == "crop":
            input_row[feature] = crop_name
        elif feature == "district":
            input_row[feature] = district if district is not None else df[feature].mode().iloc[0] if not df[feature].mode().empty else "Nagpur"

    input_df = pd.DataFrame([input_row], columns=feature_columns)
    predicted_yield = float(model.predict(input_df)[0])
    return {
        "status": "ok",
        "predicted_yield_ton_per_hectare": round(predicted_yield, 3),
        "notes": [f"Estimated yield for {crop_name.title()} under this profile is {predicted_yield:.2f} tons/hectare."],
    }


def recommend_best_crops_for_region(
    df: pd.DataFrame,
    district: str | None = None,
    climate: str | None = None,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    subset = df.copy()
    if district and "district" in subset.columns:
        district_matches = subset[subset["district"].astype(str).str.lower() == str(district).lower()]
        if not district_matches.empty:
            subset = district_matches

    if climate:
        climate_key = str(climate).lower()
        climate_filters = []
        if climate_key in {"hot", "dry", "semi-arid", "arid"}:
            climate_filters = ["Low", "Medium"]
        elif climate_key in {"humid", "wet", "tropical"}:
            climate_filters = ["High", "Medium"]
        elif climate_key in {"cool", "temperate"}:
            climate_filters = ["Low", "Medium"]
        if climate_filters and "water_availability" in subset.columns:
            subset = subset[subset["water_availability"].isin(climate_filters)]

    if subset.empty:
        return []

    crops = subset.groupby("crop").agg(
        avg_yield=("yield_ton_per_hectare", "mean") if "yield_ton_per_hectare" in subset.columns else ("crop", "count"),
        avg_soil_ph=("soil_ph", "mean") if "soil_ph" in subset.columns else ("crop", "count"),
        avg_rainfall=("rainfall_mm", "mean") if "rainfall_mm" in subset.columns else ("crop", "count"),
    ).reset_index()

    if "avg_yield" in crops.columns and crops["avg_yield"].notna().any():
        crops = crops.sort_values("avg_yield", ascending=False).head(top_n)
    else:
        crops = crops.head(top_n)

    recommendations = []
    for _, row in crops.iterrows():
        crop = row.get("crop")
        score = row.get("avg_yield", 0)
        recommendations.append({
            "crop": str(crop),
            "score": round(float(score), 2) if pd.notna(score) else 0.0,
            "avg_yield_ton_per_hectare": round(float(row.get("avg_yield", 0)), 2) if pd.notna(row.get("avg_yield")) else None,
            "avg_soil_ph": round(float(row.get("avg_soil_ph", 0)), 2) if pd.notna(row.get("avg_soil_ph")) else None,
            "avg_rainfall_mm": round(float(row.get("avg_rainfall", 0)), 2) if pd.notna(row.get("avg_rainfall")) else None,
        })
    return recommendations


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

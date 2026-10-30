import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def build_gemini_prompt(columns: List[str], sample_rows: int = 5) -> str:
    return (
        "You are a data-mapping assistant for an agriculture analytics platform. "
        "Return JSON only with an array of objects containing source_column, mapped_to, and confidence. "
        "Map each source column to the closest canonical field. "
        "Canonical fields include: crop, soil_ph, rainfall_mm, temperature_c, humidity_pct, nitrogen, phosphorus, potassium, yield_ton_per_hectare, district, season, water_availability.\n"
        f"Columns: {columns}.\n"
        f"Return only JSON, no markdown, no explanation."
    )


def build_agronomy_prompt(crop: str, conditions: Dict[str, Any], district: Optional[str] = None, climate: Optional[str] = None, yield_estimate: Optional[float] = None, suitability: Optional[str] = None, confidence: Optional[float] = None) -> str:
    return (
        "You are an agronomy advisor for a crop suitability application. Return JSON only with keys: summary, risks, recommendations, next_actions. "
        "Keep the output practical and concise.\n"
        f"Crop: {crop}\n"
        f"Conditions: {json.dumps(conditions, default=str)}\n"
        f"District: {district or 'not provided'}\n"
        f"Climate profile: {climate or 'not provided'}\n"
        f"Yield estimate (tons/ha): {yield_estimate if yield_estimate is not None else 'unknown'}\n"
        f"Suitability: {suitability or 'unknown'}\n"
        f"Confidence: {confidence if confidence is not None else 'unknown'}\n"
        "Return only valid JSON with the keys summary, risks, recommendations, next_actions."
    )


def call_gemini_for_mapping(columns: List[str]) -> Optional[List[Dict[str, Any]]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        payload = {
            "contents": [{
                "parts": [{"text": build_gemini_prompt(columns)}]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
        }
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("candidates"):
            return None
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("` ")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("mappings"), list):
            return parsed["mappings"]
        return None
    except Exception:
        return None


def generate_agronomy_explanation(
    crop: str,
    conditions: Dict[str, Any],
    district: Optional[str] = None,
    climate: Optional[str] = None,
    yield_estimate: Optional[float] = None,
    suitability: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    confidence_value = 0.0 if confidence is None else float(confidence)
    if not api_key:
        return {
            "summary": f"{crop.title()} is being evaluated under agronomic conditions that are {suitability.lower() if suitability else 'being assessed'} with a confidence of {confidence_value * 100:.1f}% if available.",
            "risks": ["Monitor pH, rainfall, temperature, and nutrient balance for the crop."],
            "recommendations": ["Keep soil pH and moisture in the crop-specific target range.", "Use split irrigation and nutrient scheduling to reduce stress."],
            "next_actions": ["Collect soil samples and compare to crop-specific agronomic guidelines.", "Review irrigation and fertilizer timing before planting or transplanting."],
        }

    try:
        payload = {
            "contents": [{"parts": [{"text": build_agronomy_prompt(crop, conditions, district, climate, yield_estimate, suitability, confidence)}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
        }
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("candidates"):
            raise ValueError("No candidates returned")
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("` ")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary") or "Agronomic review not available."),
                "risks": parsed.get("risks") or ["No major risk identified in the current conditions."],
                "recommendations": parsed.get("recommendations") or ["Continue monitoring key agronomic variables."],
                "next_actions": parsed.get("next_actions") or ["Validate with local field data."],
            }
        raise ValueError("Unexpected LLM output")
    except Exception:
        return {
            "summary": f"{crop.title()} should be assessed with field monitoring under the provided agronomic profile.",
            "risks": ["Nutrient imbalance, moisture stress, or temperature deviation could reduce yield.", "Seasonal timing should match crop-specific growth windows."],
            "recommendations": ["Keep the soil pH and water regime within target ranges for this crop.", "Schedule fertilizer and irrigation around the crop stage and forecasted weather."],
            "next_actions": ["Take soil samples for validation.", "Compare local conditions to the crop-specific agronomic thresholds before commitment."],
        }

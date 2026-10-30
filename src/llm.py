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

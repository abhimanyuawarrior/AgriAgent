from typing import Any, Dict, List

import pandas as pd
import plotly.express as px


def generate_agriculture_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "crop_count": int(df["crop"].nunique()) if "crop" in df.columns else 0,
        "avg_soil_ph": round(float(df["soil_ph"].mean()), 2) if "soil_ph" in df.columns else None,
        "avg_rainfall_mm": round(float(df["rainfall_mm"].mean()), 2) if "rainfall_mm" in df.columns else None,
        "avg_yield": round(float(df["yield_ton_per_hectare"].mean()), 2) if "yield_ton_per_hectare" in df.columns else None,
    }

    if "crop" in df.columns and "yield_ton_per_hectare" in df.columns and not df.empty:
        yield_by_crop = df.groupby("crop")["yield_ton_per_hectare"].mean()
        if not yield_by_crop.empty:
            summary["best_crop"] = yield_by_crop.sort_values(ascending=False).head(1).index[0]

    return summary


def generate_chart_collection(df: pd.DataFrame) -> Dict[str, Any]:
    charts: Dict[str, Any] = {}

    if "crop" in df.columns:
        crop_counts = df["crop"].value_counts().reset_index()
        crop_counts.columns = ["crop", "count"]
        charts["crop_distribution"] = px.bar(crop_counts, x="crop", y="count", title="Crop Distribution")

    if "soil_ph" in df.columns and "yield_ton_per_hectare" in df.columns:
        charts["yield_vs_soil"] = px.scatter(
            df,
            x="soil_ph",
            y="yield_ton_per_hectare",
            color="crop" if "crop" in df.columns else None,
            title="Yield vs Soil pH",
        )

    if "rainfall_mm" in df.columns and "yield_ton_per_hectare" in df.columns:
        charts["yield_vs_rainfall"] = px.scatter(
            df,
            x="rainfall_mm",
            y="yield_ton_per_hectare",
            color="crop" if "crop" in df.columns else None,
            title="Yield vs Rainfall",
        )

    if "district" in df.columns:
        grouped = df.groupby("district")["yield_ton_per_hectare"].mean().reset_index() if "yield_ton_per_hectare" in df.columns else df.groupby("district").size().reset_index()
        if "yield_ton_per_hectare" in df.columns:
            grouped.columns = ["district", "avg_yield"]
        else:
            grouped.columns = ["district", "count"]
        charts["regional_yield"] = px.bar(grouped, x="district", y=grouped.columns[1], title="Regional Yield")

    return charts


def generate_agri_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []

    if "yield_ton_per_hectare" in df.columns:
        avg_yield = float(df["yield_ton_per_hectare"].mean())
        insights.append(f"Average observed yield is {avg_yield:.2f} tons/hectare.")

    if "soil_ph" in df.columns:
        soil_quality = df["soil_ph"].mean()
        if soil_quality < 5.5:
            insights.append("Soil pH is on the acidic side; liming or pH correction may improve nutrient uptake.")
        elif soil_quality > 7.5:
            insights.append("Soil pH is alkaline; nutrient balancing and amendment planning may improve productivity.")
        else:
            insights.append("Soil pH is within a healthy mid-range for most crop varieties.")

    if "rainfall_mm" in df.columns:
        rainfall_avg = float(df["rainfall_mm"].mean())
        if rainfall_avg < 800:
            insights.append("Rainfall is below typical crop-support levels; irrigation planning should be prioritized.")
        else:
            insights.append("Rainfall availability appears supportive of crop growth in the current dataset.")

    if "crop" in df.columns and "yield_ton_per_hectare" in df.columns:
        best_crop = df.groupby("crop")["yield_ton_per_hectare"].mean().sort_values(ascending=False).head(1)
        if not best_crop.empty:
            insights.append(f"The highest-performing crop in the dataset is {best_crop.index[0]}.")

    if not insights:
        insights.append("Data is loaded successfully. Add more agronomic variables for deeper crop intelligence.")

    return insights

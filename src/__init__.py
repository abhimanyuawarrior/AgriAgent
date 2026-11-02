"""AgriAgent package."""

from .agent_orchestrator import (
    AgenticDataOrchestrator,
    DataCleaningAgent,
    DatasetPreparationAgent,
    prepare_uploaded_datasets,
    run_agentic_pipeline,
)

try:
    from .fao_stat import fetch_fao_stat_data, fetch_regional_crop_yield_rainfall, get_fao_stat_token
except ModuleNotFoundError:  # Optional dependency not present in this repo snapshot.
    fetch_fao_stat_data = None
    fetch_regional_crop_yield_rainfall = None
    get_fao_stat_token = None

__all__ = [
    "DataCleaningAgent",
    "DatasetPreparationAgent",
    "AgenticDataOrchestrator",
    "prepare_uploaded_datasets",
    "run_agentic_pipeline",
    "fetch_fao_stat_data",
    "fetch_regional_crop_yield_rainfall",
    "get_fao_stat_token",
]

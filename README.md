# AgriAgent

AgriAgent is an agentic AI platform for automated data engineering and intelligent analytics in agriculture. It ingests heterogeneous agricultural datasets, profiles them, maps schema fields, cleans and transforms them into a Bronze/Silver/Gold pipeline, validates quality, and generates actionable analytics.

## Features

- CSV, JSON, and Excel ingestion for single or multiple uploaded datasets
- Automatic dataset profiling and schema mapping
- Bronze, Silver, and Gold medallion layers
- Data validation and self-correction guidance
- Automated dataset cleaning and preparation agent for downstream analytics
- Agriculture analytics and crop-value insights
- Streamlit dashboard for interactive exploration

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Configure optional access keys:
   create a `.env` file and set `GEMINI_API_KEY` for LLM-based schema recommendation, and `FAOSTAT_API_TOKEN` for live FAOSTAT crop/yield/rainfall data.
4. Start the app:
   `python -m streamlit run app.py`
5. Log in with the default admin account (`admin` / `admin123`) or create a new one and upload a single or multiple CSV, JSON, or Excel datasets; the agent will clean and prepare them for downstream processing.

## Project structure

- `app.py` — Streamlit interface
- `src/data_pipeline.py` — ingestion, profiling, schema mapping, cleaning, validation
- `src/analytics.py` — agriculture analytics and charts
- `src/agent_orchestrator.py` — end-to-end agentic workflow orchestration
- `data/sample_agri_dataset.csv` — sample agricultural dataset

## Notes

The system is designed to be LLM-ready. If a `GEMINI_API_KEY` is present, the project can be extended to use a live LLM for richer semantic schema matching and transformation generation. The default prototype works without an external API using deterministic heuristics.

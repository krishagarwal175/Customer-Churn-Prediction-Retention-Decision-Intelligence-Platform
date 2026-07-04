# Copilot Instructions for This Repository

## Build, test, and lint commands

This project uses a `src/` layout (`pyproject.toml` sets `pythonpath = ["src"]` for pytest).

```bash
# Environment setup
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

```bash
# Run tests
pytest

# Run a single test file
pytest tests/test_loader.py

# Run a single test
pytest tests/test_loader.py::test_load_dataset_reads_csv
```

```bash
# Lint/format checks (tools are listed in requirements-dev.txt)
ruff check .
black --check .
isort --check-only .
```

```bash
# Run implemented pipelines
python -m churn_platform.data.ingestion --config config/config.yaml
python -m churn_platform.preprocessing.pipeline --config config/preprocessing_config.yaml
```

## High-level architecture

The implemented core is a two-stage data pipeline:

1. **Ingestion/validation stage** (`src/churn_platform/data/ingestion.py`)
   - Loads raw telco data (`load_dataset` in `data/loader.py`)
   - Validates against a centralized schema contract (`data/schema.py` + `data/validator.py`)
   - Writes validation/profile/data dictionary outputs to `reports/` and `docs/`
   - Copies validated raw data to `data/interim/` (raw data remains unchanged)

2. **Preprocessing stage** (`src/churn_platform/preprocessing/pipeline.py`)
   - Loads validated interim data
   - Cleans and normalizes with `DataCleaner`
   - Defines binary target (`Churn Label` -> 1/0)
   - Removes identifier/leakage/reporting columns via `FeatureSelector`
   - Applies column-wise sklearn transforms (one-hot + numeric scaling)
   - Produces stratified train/validation/test outputs and persists:
     - fitted preprocessor artifact
     - processed split payload
     - markdown preprocessing report

Most other domains (`models/`, `features/`, `simulation/`, `recommendations/`, Streamlit pages/components) are currently placeholders; treat `data/` + `preprocessing/` as the active production path.

## Key repository conventions

- **Config-driven behavior first:** runtime paths and preprocessing rules come from YAML under `config/` (`config.yaml`, `preprocessing_config.yaml`), not hardcoded constants.
- **Simple local YAML parsing is intentional:** ingestion/preprocessing config loaders use lightweight in-repo parsers instead of requiring full YAML tooling at runtime startup.
- **Single schema contract:** keep source-column expectations in `data/schema.py`; validation, profiling, and preprocessing assumptions should align with that file.
- **Leakage handling is explicit:** columns like `Churn Value`, `Churn Score`, `Churn Reason`, and configured reporting/identifier fields are excluded in preprocessing by configuration.
- **Backward-compatible import shims exist:** `data/loading.py` and `data/validation.py` re-export newer modules; keep them working when changing loader/validator APIs.
- **Project logging is centralized:** use `churn_platform.utils.logging.configure_logging()` for CLI pipelines to keep consistent formatting/levels.

# Customer Churn Prediction & Retention Decision Intelligence Platform

Production-style analytics platform scaffold for churn prediction, explainability, segmentation, revenue simulation, retention recommendations, and Streamlit dashboarding.

This repository currently contains the project structure only. Business logic, model training, feature engineering, and dashboard implementation are intentionally not implemented yet.

## Project Scope

The platform will use the IBM Telco Customer Churn dataset exclusively and will follow the architecture defined in:

- `outputs/customer_churn_platform_implementation_roadmap.md`

## Repository Layout

```text
config/                 Configuration files
data/                   Raw, interim, processed, and external data folders
artifacts/              Persisted models, preprocessors, metrics, and explainability outputs
notebooks/              Exploratory notebooks, if used later
reports/                Figures, model cards, and business insight artifacts
src/churn_platform/     Main Python package
app/                    Streamlit application shell
tests/                  Test suite placeholders
docs/                   Project documentation
outputs/                User-facing planning deliverables
```

## Environment Setup

Recommended local setup:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Current Status

- Git repository initialized.
- Project folder structure created.
- Dependency manifest created.
- Configuration placeholders created.
- Python package placeholders created.
- Streamlit app placeholders created.
- Test and documentation placeholders created.

## Next Implementation Phase

The next phase should implement dataset ingestion and validation only, following the milestone roadmap. Business logic should remain separated from machine learning logic and dashboard logic.


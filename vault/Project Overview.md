# Project Overview

**Goal:** Production-grade *Customer Churn Prediction & Retention Decision Intelligence Platform* — resembles an internal telecom analytics platform (not a Kaggle/edu notebook).

**Principles:** config-driven, modular, clean architecture, PEP8, type hints, Google docstrings, logging, defensive programming, no overengineering.

## Dataset
- **Newer** IBM Telco *Customer Churn Status* dataset (NOT classic Telco).
- Raw file: `data/raw/Telco_customer_churn.xlsx`.
- Column names contain **spaces**. Target = `Churn Label` (not `Churn`).
- Key cols: `Monthly Charges`, `Total Charges`, `Tenure Months`, `Contract`, service/segment cols.

## Key API contracts (do not break)
- `utils/config.py` → `load_config(path)` returns **dict**.
- `preprocessing/pipeline.py` → `PreprocessingPipeline(config, schema)`, method `transform(df)`.
- Cleaner only strips whitespace from names — it **does not rename** columns.

See [[Architecture]] · [[Status]]

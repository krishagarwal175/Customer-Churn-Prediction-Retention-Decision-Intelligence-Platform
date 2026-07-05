# Status

**As of 2026-07-05**

## Done ✅
- Scaffold, venv, git, ruff, black, config system, logging, tests.
- Data validation / loading / cleaning / preprocessing pipeline / cleaning reports.
- **BusinessEDA** — schema-driven refactor complete & verified.
- `01_business_eda.ipynb` notebook wired end-to-end.
- **FeatureEngineer** — schema-driven, 5 engineered features, verified.
- **Modeling** — training (LR + XGBoost), evaluation (+recall@precision), calibration, prediction/persistence. Dataset-agnostic, verified on real data.
- **Explainability** — SHAP global + per-customer, business-readable driver translation. Verified LR + XGB.
- Memory vault initialized.

## Health
- `ruff check .` clean · `black` formatted · `pytest` **45 passed**.

## Next 🔜
- Segmentation (clustering/personas), simulation (revenue/sensitivity), recommendations, reporting, Streamlit app. Optional: feature selection, model-selection orchestration script.

See [[Open Questions]]

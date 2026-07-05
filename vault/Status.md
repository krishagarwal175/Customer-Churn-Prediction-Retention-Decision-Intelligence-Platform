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
- **Segmentation** — K-Means clustering + persona labelling. Verified on real data (4 coherent segments).
- **Simulation** — expected-value revenue targeting (segment-specific uplift/cost) + full-grid sensitivity. Verified on real data (ROI 4.0).
- **Recommendations** — hybrid persona+driver rules, expected-net-benefit prioritization, Top-N/budget cap. Verified end-to-end.
- **Reporting** — Plotly+Matplotlib charts (figure objects) + plain-DataFrame tables for all 4 components. Verified on real data.
- Memory vault initialized.

## Health
- `ruff check .` clean · `black` formatted · `pytest` **84 passed**.

## Design decisions (user-chosen)
- Modeling stays LR + XGBoost (NN/transformer deferred to a dedicated step).
- Simulation: expected-value + segment economics + full-grid sensitivity.

## Next 🔜
- Streamlit app (UI) — WAIT for user's detailed description before building. All backend layers complete. Optional: feature selection, model-selection orchestration script.

See [[Open Questions]]

# Progress Log

### 2026-07-05 — BusinessEDA refactor + vault init
- Analyzed failure: hardcoded classic-Telco columns → `ValueError`.
- Added `schema.eda` semantic mapping to `config/config.yaml`.
- Refactored `analysis/business_eda.py` → schema-driven; fixed `generate_report` missing return.
- Built `notebooks/exploratory/01_business_eda.ipynb` exercising all 8 methods (config → pipeline.transform → BusinessEDA).
- Verified: `ruff` clean, `black` formatted, `pytest` **19 passed**, smoke test of all 8 methods OK.
- Created memory vault (`vault/`).

### 2026-07-05 — Full audit
- `ruff` clean · `black` 80 files unchanged · `pytest` **19 passed**.
- Import sweep: **49/49** modules import cleanly, no errors.
- End-to-end on real dataset: raw (7043, 33) → clean (7043, 29); churn rate **26.54%** (canonical); all 8 EDA methods return; input not mutated.
- Raw `.xlsx` present. Health green.

### 2026-07-05 — Milestone: Feature Engineering
- Implemented `features/engineering.py` → `FeatureEngineer` (schema-driven, non-mutating, computation-only), replacing placeholder.
- Produces the 5 schema-declared engineered columns: `tenure_bucket`, `service_count`, `average_monthly_spend`, `contract_commitment_score`, `risk_value_quadrant`.
- Missing-column features skip gracefully with a warning.
- Added `schema.features` config block; wrote 9 real tests (replacing placeholder).
- Verified: `ruff` clean · `black` formatted · `pytest` **28 passed** · E2E on real data adds all 5 cols, input unmutated.

### 2026-07-05 — Milestone: Modeling (train/eval/calibrate/predict)
- Implemented `models/training.py` (dtype-inferred preprocessing pipeline + config-driven estimator: logistic_regression baseline, xgboost candidate; `encode_target`).
- `models/evaluation.py` — standard metrics + primary `recall_at_precision_floor` (floor 0.45).
- `models/calibration.py` — `calibrate_classifier` (FrozenEstimator + CalibratedClassifierCV, sklearn 1.9 API).
- `models/prediction.py` — `ChurnPredictor` (proba/threshold, joblib save/load).
- 9 tests (replacing placeholder). Dataset-agnostic (roles inferred by dtype).
- Real E2E: LR ROC-AUC 0.833 / XGB 0.853; recall@p≥0.45 = 0.875 / 0.925; calibration Brier 0.159→0.146.
- Verified: `ruff` clean · `black` formatted · `pytest` **37 passed**.

### 2026-07-05 — Milestone: Explainability (SHAP)
- `explainability/shap_explainer.py` → `ChurnExplainer`: SHAP over a fitted pipeline; `global_importance` + `explain_customer`. TreeExplainer for tree models, masker-based fallback for linear (sklearn/shap 0.52).
- `explainability/business_translator.py` → `BusinessTranslator` + `humanize_feature`: turns transformed feature names into readable, directional churn drivers (categorical-only Col=Value split).
- 8 tests (new `test_explainability.py`). Verified both LR + XGB paths on real data; sensible top drivers (tenure, contract commitment, dependents).
- `ruff` clean · `black` formatted · `pytest` **45 passed**.

### 2026-07-05 — Milestone: Segmentation (clustering + personas)
- `segmentation/clustering.py` → `CustomerSegmenter` + `SegmentationConfig`: K-Means pipeline (impute+scale+kmeans), auto/config feature selection, non-mutating `assign_segments`, `profile` with optional churn rate.
- `segmentation/personas.py` → `PersonaLabeler` + `DEFAULT_PERSONA_NAMES`: value×risk median-split quadrant persona labels.
- 8 tests (`test_segmentation.py`). Real E2E: 4 coherent segments (High-Value Loyal churn 0.14, Low-Value At-Risk 0.46, etc.).
- `ruff` clean · `black` formatted · `pytest` **53 passed**.

### 2026-07-05 — Milestone: Simulation (revenue + sensitivity)
- User-chosen design: expected-value targeting, segment-specific uplift/cost, full-grid sensitivity, keep LR+XGBoost.
- `simulation/revenue.py` → `RevenueSimulator` + `RetentionEconomicsConfig`: per-customer expected saved revenue / net benefit / target flag; per-segment uplift+cost with global defaults; `campaign_summary` (ROI).
- Added `uplift_scale`/`cost_scale` global multipliers so grid sensitivity moves segment-specific economics (real-run exposed that sweeping the global default did nothing when segment overrides cover all rows).
- `simulation/sensitivity.py` → `SensitivityAnalyzer.run_grid` (full grid via dataclasses.replace over config fields).
- 11 tests. Real E2E: ROI 4.0, ~$1.14M expected net benefit on 4817 targeted customers; grid responds to scale sweeps.
- `ruff` clean · `black` formatted · `pytest` **64 passed**.

### 2026-07-05 — Milestone: Recommendations
- User-chosen design: hybrid (persona playbook + driver actions), expected-net-benefit prioritization, config-driven YAML catalog, Top-N/budget cap.
- `recommendations/rules.py` → `RetentionRuleEngine` (+`from_config`): merges persona playbook + driver-triggered actions, de-duped/ordered.
- `recommendations/prioritization.py` → `RetentionPrioritizer` + `PrioritizationConfig`: rank by expected net benefit, eligibility (net>0), optional top_n + budget cap; non-mutating.
- Added `recommendations` catalog block to `config.yaml` (personas + driver_rules).
- 10 tests. Real E2E: prioritized top-5 targets with sensible hybrid actions.
- `ruff` clean · `black` formatted · `pytest` **74 passed**. Full platform pipeline now integrated end-to-end.

_Next iteration: append here._

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

_Next iteration: append here._

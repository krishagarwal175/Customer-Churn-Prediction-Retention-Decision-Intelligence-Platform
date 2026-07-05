# Decision Log

### D1 — BusinessEDA is schema-driven, not hardcoded (2026-07-05)
- **Why:** It hardcoded classic-Telco names (`Churn`, `MonthlyCharges`…) → `ValueError` on the active dataset; violated the config-driven philosophy.
- **What:** Resolve all columns from a semantic mapping. `DEFAULT_EDA_SCHEMA` module constant matches the active dataset so `BusinessEDA(df)` still works; optional `schema` arg injects overrides.

### D2 — Minimal config extension: `schema.eda` (2026-07-05)
- **Why:** Existing `schema` block had no semantic mapping for charges/tenure/contract/services/segments.
- **What:** Added additive `schema.eda` mapping (logical role → column name + `positive_churn_value`, `month_to_month_value`). Purely additive; cleaner ignores unknown keys, so nothing else breaks.

### D3 — Backward-compatible constructor
- Kept `BusinessEDA(clean_df)` single-arg working (defaults). Notebook passes `schema=config["schema"]["eda"]` to demonstrate config injection.

### D4 — Scope discipline
- Refactored **only** `analysis/business_eda.py` + additive config. No unrelated modules touched. Also fixed latent bug: `generate_report()` never returned its dict.

### D5 — FeatureEngineer mirrors BusinessEDA conventions (2026-07-05)
- **Why:** Feature layer was a placeholder; needed the schema-declared engineered columns while staying config-driven.
- **What:** `FeatureEngineer(schema=None)`, `transform(df)` returns new df (non-mutating, computation-only). Features: `tenure_bucket`, `service_count`, `average_monthly_spend` (Total/tenure, zero-tenure → Monthly), `contract_commitment_score` (ordinal), `risk_value_quadrant` (CLTV median × contract-commitment risk). Missing sources skip with a warning. Added additive `schema.features` config.

See [[Progress Log]]

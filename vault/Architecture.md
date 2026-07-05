# Architecture

```
src/churn_platform/
  analysis/      business_eda.py (computation-only EDA), plotting.py
  data/          loading, validation, schema, profiling, ingestion
  preprocessing/ cleaning.py (DataCleaner), pipeline.py (PreprocessingPipeline)
  features/ models/ explainability/ segmentation/ simulation/ recommendations/ reporting/
  utils/         config.py (load_config), logging, paths, constants
config/  data/  tests/  notebooks/  docs/  artifacts/  reports/  work/  vault/
```

## BusinessEDA (analysis, computation-only)
- **Schema-driven**: resolves every business column from a semantic mapping, never hardcodes names.
- Constructor: `BusinessEDA(data, schema=None)` — `schema` overrides `DEFAULT_EDA_SCHEMA` (typically `config["schema"]["eda"]`).
- Contracts: defensive copy, never mutates input, validates columns, returns dict/DataFrame, no side effects (no plotting/IO/ML).
- Public API: `churn_summary`, `revenue_summary`, `contract_analysis`, `tenure_analysis`, `service_analysis`, `customer_segmentation`, `executive_dashboard`, `generate_report`.

## Config schema (`config/config.yaml` → `schema`)
- `target_column`, `identifier_columns`, `leakage_columns`, `required_columns` (used by cleaner).
- **`schema.eda`** (added) — semantic mapping consumed by BusinessEDA only.

See [[Decisions]]

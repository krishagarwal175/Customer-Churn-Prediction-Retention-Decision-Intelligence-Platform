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

### 2026-07-05 — Milestone: Reporting (charts + tables)
- User-chosen design: both Plotly + Matplotlib, figure objects (no IO), plain DataFrames, all 4 components.
- `reporting/charts.py`: `churn_overview`, `revenue_at_risk`, `roc_curve`, `precision_recall`, `calibration`, `feature_importance` — each with `backend="plotly"|"matplotlib"`, returns a figure, no side effects.
- `reporting/tables.py`: `kpi_overview_table`, `model_metrics_table`, `segment_profile_table`, `retention_shortlist_table` — plain DataFrames.
- 10 tests. Real E2E: all charts (both backends) + tables built from the live model.
- `ruff` clean · `black` formatted · `pytest` **84 passed**.
- Last backend layer done. UI (Streamlit) is next — WAIT for user's detailed UI description before building.

### 2026-07-05 — Milestone: Streamlit UI (v1)
- User: build a first UI pass and iterate. Theme locked: Nordic Steel light + Graphite dark (toggle).
- Built `app/`: `theme.py` (palettes/CSS/plotly styling/toggle), `services.py` (cached pipeline: data, model, segmentation, scoring, simulation, recommendations), components (kpis/charts/tables/filters/customer_profile), 8 views, `streamlit_app.py` (st.navigation). `.streamlit/config.toml`, `.claude/launch.json`.
- Renamed `app/pages`→`app/views` to stop Streamlit auto-MPA clashing with st.navigation. Fixed unique `url_path`, `st.dataframe` height=None, `use_container_width`→`width`, zero-baseline charts.
- Verified live: exec dashboard, prediction, revenue simulator render in both light+dark; no errors. Backend `pytest` still **84 passed**.
- KNOWN ITERATION ITEMS: (1) model uses geographic/reporting cols (Zip Code, Lat/Long) as features → "Zip Code" shows as a churn driver; drop reporting cols from feature set for credibility. (2) st.metric truncates long persona/contract text. (3) dark-mode native-widget polish.

### 2026-07-05 — Pivot: FastAPI → Vercel (Streamlit removed)
- User scrapped Streamlit; new target = FastAPI on Vercel. Also declined a fake-commit bot (deceptive) — see [[Decisions]].
- Design (user-chosen): precompute-and-serve + live `/predict` for new customers; REST API + minimal HTML page.
- Removed `app/`, `.streamlit/`. Split deps: slim runtime `requirements.txt` (fastapi/pydantic/numpy/pandas/jinja2) vs heavy offline `requirements-dev.txt`.
- `scripts/build_artifacts.py`: offline builder → slim JSON artifacts (kpis, metrics, customers, segments, drivers, recommendations, segmentation, catalog) + NumPy-only `model.json`. Dropped geographic/reporting cols from the model (fixes ZIP-as-driver; ROC-AUC held 0.85).
- `api/`: `data.py` (artifact loader + pure-python simulation), `predictor.py` (NumPy-only live scoring: FE + logistic + drivers + persona + hybrid recs), `schemas.py`, `index.py` (endpoints + themed HTML landing page), `templates/index.html`.
- `vercel.json` rewrites; `.claude/launch.json` uvicorn.
- Verified live in browser: landing page + live-score form work (86.9% high-risk demo). `ruff`/`black` clean, `pytest` **95 passed** (84 backend + 11 API).
- REMAINING: user connects repo to Vercel to deploy (needs their Vercel account); heavy libs never ship to Vercel.

### 2026-07-05 — UI overhaul (dark/green) + enriched metrics
- User: "complete overhaul" — darker blacks, deeper green, loading animations, better accuracy ratios.
- `scripts/build_artifacts.py`: `_enriched_metrics` reports at the F1-optimal threshold → accuracy 0.785, recall 0.751, precision 0.573, f1 0.650 (was recall 0.32 at 0.5); kept roc_auc 0.849, recall@precision 0.865. Regenerated metrics.json.
- Rebuilt `api/templates/index.html`: near-black (#050607) + deep emerald (#0B7A57/#10B981/#34D399) theme; sidebar nav w/ scroll-spy; full-screen loader, count-up numbers, SVG ring gauges, animated bars, section fade-ins, shimmer skeletons; live ROI simulator (sliders→/api/simulate) and live scoring form (→/api/predict) with animated probability ring.
- Verified in browser: all sections populate, simulator live (ROI 6.4×), live scoring 87% high-risk. `pytest` **95 passed**.
- Auto-deploys to Vercel on push (churn-decision-intelligence). NOTE: still behind Vercel deployment protection (SSO) until user disables it.

### 2026-07-09 — Brutalist editorial redesign + deploy
- Rebuilt `api/templates/index.html` per julianejeske-inspired brief: Syne/Space Grotesk/Space Mono, near-black charcoal + acid-green (#C7F94E) accent + red (#FF4A34) risk, 1px dividers, bracketed `[01]`/slash labels, massive hero + live risk simulator, scroll marquee, numbered process steps, inverse-hover, cubic-bezier(.16,1,.3,1) reveals. API wiring retained.
- Verified in browser (hero sim 87%, all sections, live scoring). `pytest` API 11 passed. Deployed to Vercel prod.
- Earlier: disabled Vercel deployment protection (ssoProtection→None) via API so the site + /api/predict are public. Gave user hi/lo test profiles.
- ⚠️ DEPLOY GOTCHA: the clean alias `churn-decision-intelligence.vercel.app` is a MANUAL alias — `vercel --prod` does NOT auto-repoint it. After every prod deploy run: `vercel alias set <new-deployment-url> churn-decision-intelligence.vercel.app`.

### 2026-07-09 — Footer credit, field tooltips, responsive fixes
- Footer: "built by Krish Agarwal" → github.com/krishagarwal175 + repo link.
- Live-scoring: info (ⓘ) icon per field with a tooltip explaining the parameter + churn effect. Hover (desktop) + click/tap toggle (`.info.open`, JS) for touch; `:focus` also wired. NOTE: preview Chrome can't test `:hover`/`:focus` (window lacks OS focus → `:focus` won't match even when activeElement) — verified via the click toggle instead.
- Responsive: `.tscroll` wrapper on the table (scrolls in-container), viewport-safe tooltip placement (nth-child edge flips), small-screen type/padding block. No horizontal overflow at 375px.
- Deployed + re-pointed alias per the gotcha. Verified live.

_Next iteration: append here._

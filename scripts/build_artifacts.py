"""Offline builder for the FastAPI runtime artifacts.

Runs the full analytics stack (XGBoost, SHAP, K-Means) once and writes slim JSON
artifacts that the serverless API serves without any heavy dependencies:

- ``kpis.json``            headline churn and revenue KPIs
- ``metrics.json``        model evaluation metrics
- ``customers.json``      scored customers (probability, segment, persona)
- ``segments.json``       segment profiles and personas
- ``drivers.json``        global SHAP feature importances
- ``recommendations.json`` prioritized retention shortlist with actions
- ``model.json``          NumPy-only logistic model + feature-engineering spec
- ``segmentation.json``   K-Means centroids for live persona assignment

Run:  python scripts/build_artifacts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn_platform.analysis.business_eda import BusinessEDA
from churn_platform.explainability.business_translator import BusinessTranslator
from churn_platform.explainability.shap_explainer import ChurnExplainer
from churn_platform.features.engineering import FeatureEngineer
from churn_platform.models.calibration import calibrate_classifier
from churn_platform.models.evaluation import evaluate_classifier
from churn_platform.models.training import (
    ModelConfig,
    build_model_pipeline,
    encode_target,
    train_model,
)
from churn_platform.preprocessing.pipeline import PreprocessingPipeline
from churn_platform.preprocessing.preprocessing_config import SplitConfig
from churn_platform.preprocessing.splitting import split_train_validation_test
from churn_platform.recommendations.prioritization import (
    PrioritizationConfig,
    RetentionPrioritizer,
)
from churn_platform.recommendations.rules import RetentionRuleEngine
from churn_platform.segmentation.clustering import CustomerSegmenter, SegmentationConfig
from churn_platform.segmentation.personas import PersonaLabeler
from churn_platform.simulation.revenue import RetentionEconomicsConfig, RevenueSimulator
from churn_platform.utils.config import load_config

CONFIG_PATH = "config/config.yaml"
ARTIFACT_DIR = Path("api/artifacts")
TARGET = "Churn Label"
# Geographic/reporting columns excluded from the model: noisy, low-signal, and
# awkward to require for live scoring. Dropping them yields cleaner drivers.
DROP_COLUMNS = [
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]
SEGMENT_FEATURES = ("Tenure Months", "Monthly Charges", "CLTV")
CUSTOMER_COLUMNS = [
    "CustomerID",
    "churn_probability",
    "segment",
    "persona",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Contract",
    "Churn Label",
]


def _to_native(value: Any) -> Any:
    """Convert NumPy scalars/arrays to JSON-native types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write(name: str, payload: Any) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, default=_to_native, indent=2), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def _export_linear_model(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Fit a logistic pipeline and export a NumPy-only inference spec."""
    pipeline = build_model_pipeline(x_train, ModelConfig(name="logistic_regression"))
    pipeline.fit(x_train, y_train)
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    numeric_cols = list(pre.transformers_[0][2])
    categorical_cols = list(pre.transformers_[1][2])
    numeric_pipe = pre.named_transformers_["numeric"]
    categorical_pipe = pre.named_transformers_["categorical"]

    scaler = numeric_pipe.named_steps["scale"]
    ohe = categorical_pipe.named_steps["encode"]
    feature_names = list(pre.get_feature_names_out())

    return {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "numeric_impute": dict(
            zip(numeric_cols, numeric_pipe.named_steps["impute"].statistics_.tolist())
        ),
        "numeric_mean": dict(zip(numeric_cols, scaler.mean_.tolist())),
        "numeric_scale": dict(zip(numeric_cols, scaler.scale_.tolist())),
        "categorical_impute": dict(
            zip(
                categorical_cols,
                [str(v) for v in categorical_pipe.named_steps["impute"].statistics_],
            )
        ),
        "categorical_categories": {
            col: [str(c) for c in cats]
            for col, cats in zip(categorical_cols, ohe.categories_)
        },
        "coef": dict(zip(feature_names, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
    }


def main() -> None:
    config = load_config(CONFIG_PATH)
    raw = pd.read_excel(config["paths"]["raw_dataset_file"])
    clean = PreprocessingPipeline(
        config=config["runtime"], schema=config["schema"]
    ).transform(raw)
    features = FeatureEngineer(schema=config["schema"].get("features")).transform(clean)
    customer_ids = pd.Series(
        raw.loc[features.index, "CustomerID"].to_numpy(), index=features.index
    )

    y = encode_target(features[TARGET])
    drop = [c for c in [TARGET, *DROP_COLUMNS] if c in features.columns]
    X = features.drop(columns=drop)

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_validation_test(
        X, y, SplitConfig()
    )

    # High-quality model for probabilities, metrics, and global drivers.
    base = train_model(x_train, y_train, ModelConfig(name="xgboost_classifier"))
    calibrated = calibrate_classifier(base, x_val, y_val)
    metrics = {
        k: round(v, 4)
        for k, v in evaluate_classifier(calibrated, x_test, y_test).items()
    }

    explainer = ChurnExplainer(base, background=x_train.sample(200, random_state=42))
    importance = explainer.global_importance(
        x_test.sample(min(400, len(x_test)), random_state=1)
    ).head(15)
    translator = BusinessTranslator()
    drivers = [
        {
            "feature": translator_humanize(row.feature),
            "importance": round(row.mean_abs_shap, 5),
        }
        for row in importance.itertuples()
    ]

    # Segmentation + personas.
    segmenter = CustomerSegmenter(
        SegmentationConfig(
            n_clusters=4, feature_columns=SEGMENT_FEATURES, churn_column=TARGET
        )
    ).fit(features)
    profile = PersonaLabeler("Monthly Charges", "churn_rate").label(
        segmenter.profile(features)
    )
    seg_to_persona = profile["persona"].to_dict()

    # Scored customer frame.
    scored = features.copy()
    scored["CustomerID"] = customer_ids.to_numpy()
    scored["churn_probability"] = calibrated.predict_proba(X)[:, 1]
    scored["segment"] = segmenter.predict(X).to_numpy()
    scored["persona"] = scored["segment"].map(seg_to_persona)

    # KPIs.
    eda = BusinessEDA(clean, schema=config["schema"].get("eda"))
    churn = eda.churn_summary()
    revenue = eda.revenue_summary()

    # Recommendations shortlist (top 50).
    engine = RetentionRuleEngine.from_config(config)
    simulated = RevenueSimulator(
        RetentionEconomicsConfig(default_uplift=0.30, default_cost=60.0)
    ).simulate(scored)
    prioritized = RetentionPrioritizer(PrioritizationConfig(top_n=50)).prioritize(
        simulated
    )
    top = prioritized[prioritized["selected"]].sort_values("priority_rank")
    recommendations = []
    for idx, row in top.iterrows():
        row_drivers = [
            d["readable"]
            for d in translator.translate(explainer.explain_customer(X.loc[[idx]]), 3)
        ]
        recommendations.append(
            {
                "rank": int(row["priority_rank"]),
                "customer_id": str(scored.loc[idx, "CustomerID"]),
                "persona": row["persona"],
                "churn_probability": round(
                    float(scored.loc[idx, "churn_probability"]), 4
                ),
                "expected_net_benefit": round(float(row["expected_net_benefit"]), 2),
                "top_drivers": row_drivers,
                "recommended_actions": engine.recommend(
                    str(row["persona"]), row_drivers
                ),
            }
        )

    # Segmentation spec for live persona assignment.
    seg_scaler = segmenter._pipeline.named_steps["scale"]  # noqa: SLF001
    seg_imputer = segmenter._pipeline.named_steps["impute"]  # noqa: SLF001
    kmeans = segmenter._pipeline.named_steps["kmeans"]  # noqa: SLF001
    segmentation_spec = {
        "features": list(SEGMENT_FEATURES),
        "impute": dict(zip(SEGMENT_FEATURES, seg_imputer.statistics_.tolist())),
        "mean": dict(zip(SEGMENT_FEATURES, seg_scaler.mean_.tolist())),
        "scale": dict(zip(SEGMENT_FEATURES, seg_scaler.scale_.tolist())),
        "centroids": kmeans.cluster_centers_.tolist(),
        "seg_to_persona": {int(k): v for k, v in seg_to_persona.items()},
    }

    # Feature-engineering constants for live scoring.
    fe = config["schema"]["features"]
    feature_engineering = {
        "tenure_column": fe["tenure_column"],
        "contract_column": fe["contract_column"],
        "monthly_charges_column": fe["monthly_charges_column"],
        "total_charges_column": fe["total_charges_column"],
        "cltv_column": fe["cltv_column"],
        "service_columns": fe["service_columns"],
        "non_service_values": fe["non_service_values"],
        "contract_commitment_scores": fe["contract_commitment_scores"],
        "tenure_bucket_edges": [-1, 12, 24, 48, 60, 72],
        "tenure_bucket_labels": ["0-12", "13-24", "25-48", "49-60", "61-72"],
        "cltv_median": float(pd.to_numeric(features["CLTV"], errors="coerce").median()),
        "max_commitment": max(fe["contract_commitment_scores"].values()),
    }

    model_spec = _export_linear_model(x_train, y_train)
    model_spec["feature_engineering"] = feature_engineering
    model_spec["input_columns"] = [
        c for c in clean.columns if c not in (*DROP_COLUMNS, TARGET, "CustomerID")
    ]

    _write("kpis.json", {"churn": churn, "revenue": revenue})
    _write("metrics.json", metrics)
    _write("drivers.json", drivers)
    _write("segments.json", profile.reset_index().to_dict(orient="records"))
    _write(
        "customers.json",
        scored[[c for c in CUSTOMER_COLUMNS if c in scored.columns]]
        .round({"churn_probability": 4})
        .to_dict(orient="records"),
    )
    _write("recommendations.json", recommendations)
    _write("segmentation.json", segmentation_spec)
    _write("model.json", model_spec)
    print("Artifacts built successfully.")


def translator_humanize(feature: str) -> str:
    """Readable feature label (thin wrapper to avoid importing UI code)."""
    from churn_platform.explainability.business_translator import humanize_feature

    return humanize_feature(feature)


if __name__ == "__main__":
    main()

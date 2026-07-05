"""Cached data and model services backing the dashboard.

Wraps the churn_platform pipeline behind Streamlit caches so the model trains
once per session and pages share prepared data. The UI holds no business logic;
it only orchestrates these services.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from churn_platform.analysis.business_eda import BusinessEDA
from churn_platform.explainability.business_translator import BusinessTranslator
from churn_platform.explainability.shap_explainer import ChurnExplainer
from churn_platform.features.engineering import FeatureEngineer
from churn_platform.models.calibration import calibrate_classifier
from churn_platform.models.evaluation import evaluate_classifier
from churn_platform.models.training import ModelConfig, encode_target, train_model
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
from churn_platform.simulation.sensitivity import SensitivityAnalyzer
from churn_platform.utils.config import load_config

CONFIG_PATH = "config/config.yaml"
_TARGET = "Churn Label"
_SEGMENT_FEATURES = ("Tenure Months", "Monthly Charges", "CLTV")


@st.cache_data(show_spinner=False)
def get_config() -> dict[str, Any]:
    """Load the project configuration."""
    return load_config(CONFIG_PATH)


@st.cache_data(show_spinner="Preparing data…")
def get_prepared_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return cleaned data, engineered features, and aligned customer ids."""
    config = get_config()
    raw = pd.read_excel(config["paths"]["raw_dataset_file"])
    clean = PreprocessingPipeline(
        config=config["runtime"], schema=config["schema"]
    ).transform(raw)
    features = FeatureEngineer(schema=config["schema"].get("features")).transform(clean)
    customer_ids = pd.Series(
        raw.loc[features.index, "CustomerID"].to_numpy(),
        index=features.index,
        name="CustomerID",
    )
    return clean, features, customer_ids


@st.cache_resource(show_spinner="Training churn model…")
def get_model_bundle() -> dict[str, Any]:
    """Train the model once and return it with splits, metrics, and explainer."""
    _, features, _ = get_prepared_data()
    y = encode_target(features[_TARGET])
    X = features.drop(columns=[_TARGET])
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_validation_test(
        X, y, SplitConfig()
    )
    base = train_model(x_train, y_train, ModelConfig(name="xgboost_classifier"))
    model = calibrate_classifier(base, x_val, y_val)
    metrics = evaluate_classifier(model, x_test, y_test)
    explainer = ChurnExplainer(
        base, background=x_train.sample(min(200, len(x_train)), random_state=42)
    )
    return {
        "base": base,
        "model": model,
        "X": X,
        "y": y,
        "x_test": x_test,
        "y_test": y_test,
        "metrics": metrics,
        "explainer": explainer,
    }


@st.cache_resource(show_spinner="Segmenting customers…")
def get_segmentation() -> dict[str, Any]:
    """Fit segmentation and return the segmenter, profile, and persona map."""
    _, features, _ = get_prepared_data()
    segmenter = CustomerSegmenter(
        SegmentationConfig(
            n_clusters=4,
            feature_columns=_SEGMENT_FEATURES,
            churn_column=_TARGET,
        )
    ).fit(features)
    profile = PersonaLabeler("Monthly Charges", "churn_rate").label(
        segmenter.profile(features)
    )
    return {
        "segmenter": segmenter,
        "profile": profile,
        "seg_to_persona": profile["persona"].to_dict(),
    }


@st.cache_data(show_spinner="Scoring customers…")
def get_scored_customers() -> pd.DataFrame:
    """Return a customer-level frame with churn probability, segment, persona."""
    _, features, customer_ids = get_prepared_data()
    bundle = get_model_bundle()
    segmentation = get_segmentation()

    X = bundle["X"]
    scored = features.copy()
    scored.insert(0, "CustomerID", customer_ids)
    scored["churn_probability"] = bundle["model"].predict_proba(X)[:, 1]
    scored["segment"] = segmentation["segmenter"].predict(X).to_numpy()
    scored["persona"] = scored["segment"].map(segmentation["seg_to_persona"])
    return scored


def get_business_eda() -> BusinessEDA:
    """Return a BusinessEDA analyzer over the cleaned data."""
    config = get_config()
    clean, _, _ = get_prepared_data()
    return BusinessEDA(clean, schema=config["schema"].get("eda"))


def simulate(
    scored: pd.DataFrame,
    default_uplift: float,
    default_cost: float,
) -> pd.DataFrame:
    """Run the revenue simulation on a scored customer frame."""
    config = RetentionEconomicsConfig(
        default_uplift=default_uplift, default_cost=default_cost
    )
    return RevenueSimulator(config).simulate(scored)


def sensitivity_grid(
    scored: pd.DataFrame, uplifts: list[float], costs: list[float]
) -> pd.DataFrame:
    """Return a full-grid sensitivity sweep over uplift and cost."""
    return SensitivityAnalyzer(scored).run_grid(
        {"default_uplift": uplifts, "default_cost": costs}
    )


def recommendation_shortlist(simulated: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build the prioritized shortlist with hybrid actions and readable drivers."""
    config = get_config()
    bundle = get_model_bundle()
    engine = RetentionRuleEngine.from_config(config)
    translator = BusinessTranslator()

    prioritized = RetentionPrioritizer(PrioritizationConfig(top_n=top_n)).prioritize(
        simulated
    )
    selected = prioritized[prioritized["selected"]].sort_values("priority_rank")

    feature_columns = list(bundle["X"].columns)
    rows = []
    for idx, row in selected.iterrows():
        drivers = [
            d["readable"]
            for d in translator.translate(
                bundle["explainer"].explain_customer(
                    bundle["X"].loc[[idx], feature_columns]
                ),
                top_n=3,
            )
        ]
        actions = engine.recommend(str(row.get("persona", "")), drivers)
        rows.append(
            {
                "rank": int(row["priority_rank"]),
                "CustomerID": row.get("CustomerID", idx),
                "persona": row.get("persona", ""),
                "churn_probability": row.get("churn_probability", float("nan")),
                "expected_net_benefit": row["expected_net_benefit"],
                "top_drivers": ", ".join(drivers),
                "recommended_actions": "; ".join(actions),
            }
        )
    return pd.DataFrame(rows)

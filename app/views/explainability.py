"""Model explainability page."""

from __future__ import annotations

import streamlit as st

from app import services
from app.components import charts, tables
from app.components.filters import header
from churn_platform.reporting.tables import model_metrics_table


def render() -> None:
    """Render global model drivers and performance metrics."""
    header(
        "Explainability",
        "What drives churn across the customer base, and how the model performs.",
        eyebrow="Explain",
    )

    bundle = services.get_model_bundle()
    sample = bundle["x_test"].sample(min(300, len(bundle["x_test"])), random_state=1)
    importance = bundle["explainer"].global_importance(sample)

    charts.feature_importance(importance, top_n=12)

    st.markdown(
        '<div class="section-title">Model performance</div>', unsafe_allow_html=True
    )
    y_score = bundle["model"].predict_proba(bundle["x_test"])[:, 1]
    left, right = st.columns(2)
    with left:
        charts.roc(bundle["y_test"], y_score)
    with right:
        charts.precision_recall(bundle["y_test"], y_score)
    charts.calibration(bundle["y_test"], y_score)

    metrics = {k: round(v, 4) for k, v in bundle["metrics"].items()}
    tables.dataframe(model_metrics_table(metrics))

"""Tests for customer segmentation and persona labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_platform.segmentation.clustering import (
    CustomerSegmenter,
    SegmentationConfig,
)
from churn_platform.segmentation.personas import PersonaLabeler


@pytest.fixture
def customers() -> pd.DataFrame:
    """Two well-separated blobs of customers with a churn column."""
    rng = np.random.default_rng(7)
    low = pd.DataFrame(
        {
            "Tenure Months": rng.integers(0, 10, size=50),
            "Monthly Charges": rng.uniform(80, 120, size=50),
            "Churn Label": rng.choice(["Yes", "No"], size=50, p=[0.6, 0.4]),
        }
    )
    high = pd.DataFrame(
        {
            "Tenure Months": rng.integers(50, 72, size=50),
            "Monthly Charges": rng.uniform(20, 50, size=50),
            "Churn Label": rng.choice(["Yes", "No"], size=50, p=[0.1, 0.9]),
        }
    )
    return pd.concat([low, high], ignore_index=True)


def test_fit_predict_assigns_expected_cluster_count(customers) -> None:
    segmenter = CustomerSegmenter(SegmentationConfig(n_clusters=2))
    labels = segmenter.fit_predict(customers[["Tenure Months", "Monthly Charges"]])
    assert labels.name == "segment"
    assert set(labels.unique()) == {0, 1}
    assert len(labels) == len(customers)


def test_assign_segments_is_non_mutating(customers) -> None:
    segmenter = CustomerSegmenter(SegmentationConfig(n_clusters=2))
    features = customers[["Tenure Months", "Monthly Charges"]]
    original = features.copy(deep=True)
    segmenter.fit(features)
    out = segmenter.assign_segments(features)
    assert "segment" in out.columns
    pd.testing.assert_frame_equal(features, original)


def test_profile_includes_churn_rate(customers) -> None:
    config = SegmentationConfig(
        n_clusters=2,
        feature_columns=("Tenure Months", "Monthly Charges"),
        churn_column="Churn Label",
    )
    segmenter = CustomerSegmenter(config).fit(customers)
    profile = segmenter.profile(customers)

    assert "customer_count" in profile.columns
    assert "churn_rate" in profile.columns
    assert profile["customer_count"].sum() == len(customers)
    assert (profile["churn_rate"].between(0, 1)).all()


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        CustomerSegmenter().predict(pd.DataFrame({"a": [1, 2, 3, 4]}))


def test_fit_requires_enough_rows() -> None:
    with pytest.raises(ValueError):
        CustomerSegmenter(SegmentationConfig(n_clusters=4)).fit(
            pd.DataFrame({"a": [1.0, 2.0]})
        )


def test_missing_configured_features_raises(customers) -> None:
    config = SegmentationConfig(n_clusters=2, feature_columns=("Nonexistent",))
    with pytest.raises(ValueError):
        CustomerSegmenter(config).fit(customers)


def test_persona_labeler_quadrants() -> None:
    profile = pd.DataFrame(
        {
            "Monthly Charges": [100.0, 100.0, 20.0, 20.0],
            "churn_rate": [0.6, 0.1, 0.6, 0.1],
        }
    )
    labelled = PersonaLabeler("Monthly Charges", "churn_rate").label(profile)
    assert labelled.loc[0, "persona"] == "High-Value At-Risk"
    assert labelled.loc[1, "persona"] == "High-Value Loyal"
    assert labelled.loc[2, "persona"] == "Low-Value At-Risk"
    assert labelled.loc[3, "persona"] == "Low-Value Stable"


def test_persona_labeler_validates_columns() -> None:
    with pytest.raises(ValueError):
        PersonaLabeler("value", "risk").label(pd.DataFrame({"value": [1, 2]}))

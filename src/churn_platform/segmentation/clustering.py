"""Unsupervised customer segmentation via K-Means.

Groups customers into behavioural segments on a configurable set of numeric
features. Feature roles default to auto-detected numeric columns so the
segmenter adapts across similar datasets; segment profiling optionally reports
churn rate when a churn column is supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

__all__ = ["SegmentationConfig", "CustomerSegmenter"]

_SEGMENT_COLUMN = "segment"


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for customer segmentation."""

    n_clusters: int = 4
    random_state: int = 42
    feature_columns: tuple[str, ...] | None = None
    churn_column: str | None = None
    positive_churn_value: str = "Yes"


class CustomerSegmenter:
    """Fit and apply K-Means customer segments."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        """Initialize the segmenter.

        Args:
            config: Segmentation configuration; defaults to a 4-cluster setup.
        """
        self.config = config or SegmentationConfig()
        self.logger = logging.getLogger(__name__)
        self._pipeline: Pipeline | None = None
        self._feature_columns: list[str] = []

    def _resolve_features(self, X: pd.DataFrame) -> list[str]:
        """Determine which columns to cluster on."""
        if self.config.feature_columns is not None:
            missing = [c for c in self.config.feature_columns if c not in X.columns]
            if missing:
                raise ValueError(f"Configured feature columns missing: {missing}")
            return list(self.config.feature_columns)

        numeric = X.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise ValueError("No numeric feature columns available for clustering.")
        return numeric

    def fit(self, X: pd.DataFrame) -> CustomerSegmenter:
        """Fit the segmentation model.

        Args:
            X: Customer feature frame.

        Returns:
            The fitted segmenter (for chaining).

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If ``X`` has fewer rows than clusters.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("`X` must be a pandas DataFrame.")
        if len(X) < self.config.n_clusters:
            raise ValueError(
                f"Need at least {self.config.n_clusters} rows to form clusters."
            )

        self._feature_columns = self._resolve_features(X)
        self._pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "kmeans",
                    KMeans(
                        n_clusters=self.config.n_clusters,
                        random_state=self.config.random_state,
                        n_init=10,
                    ),
                ),
            ]
        )
        self._pipeline.fit(X[self._feature_columns])
        self.logger.info(
            "Fitted %d segments on %d rows, %d features.",
            self.config.n_clusters,
            len(X),
            len(self._feature_columns),
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Assign segment labels to customers.

        Args:
            X: Customer feature frame.

        Returns:
            Integer segment labels indexed like ``X``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._pipeline is None:
            raise RuntimeError("Segmenter must be fitted before predicting.")
        labels = self._pipeline.predict(X[self._feature_columns])
        return pd.Series(labels, index=X.index, name=_SEGMENT_COLUMN)

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """Fit the model and return segment labels for ``X``."""
        return self.fit(X).predict(X)

    def assign_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with an added ``segment`` column.

        Args:
            df: Customer frame (must contain the fitted feature columns).

        Returns:
            New DataFrame with a ``segment`` column; input is not mutated.
        """
        result = df.copy(deep=True)
        result[_SEGMENT_COLUMN] = self.predict(df).to_numpy()
        return result

    def profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize each segment's size, feature means, and churn rate.

        Args:
            df: Customer frame containing the fitted feature columns and,
                optionally, the configured churn column.

        Returns:
            DataFrame indexed by segment with ``customer_count``, mean of each
            clustering feature, and ``churn_rate`` when a churn column is set.
        """
        labelled = self.assign_segments(df)
        grouped = labelled.groupby(_SEGMENT_COLUMN)

        profile = grouped[self._feature_columns].mean()
        profile.insert(0, "customer_count", grouped.size().astype(int))

        churn_column = self.config.churn_column
        if churn_column and churn_column in labelled.columns:
            positive = self.config.positive_churn_value.strip().lower()
            is_churned = (
                labelled[churn_column].astype(str).str.strip().str.lower() == positive
            )
            profile["churn_rate"] = is_churned.groupby(labelled[_SEGMENT_COLUMN]).mean()

        return profile

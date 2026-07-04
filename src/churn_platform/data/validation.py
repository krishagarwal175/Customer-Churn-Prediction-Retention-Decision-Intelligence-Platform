"""Backward-compatible import location for dataset validation utilities."""

from churn_platform.data.validator import (
    ValidationIssue,
    ValidationResult,
    save_validation_result,
    validate_dataset,
)

__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "save_validation_result",
    "validate_dataset",
]

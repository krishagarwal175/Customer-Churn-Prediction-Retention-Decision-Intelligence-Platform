# Preprocessing Report

## Target Definition

- Target column: `Churn Label`
- Positive class: `Yes` mapped to `1`
- Negative class: `No` mapped to `0`
- Duplicate target columns are excluded from model features.

## Features Retained

- `City`
- `Zip Code`
- `Latitude`
- `Longitude`
- `Gender`
- `Senior Citizen`
- `Partner`
- `Dependents`
- `Tenure Months`
- `Phone Service`
- `Multiple Lines`
- `Internet Service`
- `Online Security`
- `Online Backup`
- `Device Protection`
- `Tech Support`
- `Streaming TV`
- `Streaming Movies`
- `Contract`
- `Paperless Billing`
- `Payment Method`
- `Monthly Charges`
- `Total Charges`
- `CLTV`

## Features Removed

| Column | Reason |
|---|---|
| CustomerID | Identifier column; useful for lookup but not generalizable as a model feature. |
| Count | Reporting-only field excluded by configuration. |
| Country | Reporting-only field excluded by configuration. |
| State | Reporting-only field excluded by configuration. |
| Lat Long | Reporting-only field excluded by configuration. |
| Churn Label | Selected target variable; used as label, not as model input. |
| Churn Value | Target leakage or post-event field; excluded from training features. |
| Churn Score | Target leakage or post-event field; excluded from training features. |
| Churn Reason | Target leakage or post-event field; excluded from training features. |

## Missing Value Handling

| Column Group | Strategy | Rationale |
|---|---|---|
| Text/categorical columns | Trim whitespace, convert empty strings to missing, fill with configured category | Preserves rows and supports unseen inference records |
| `Total Charges` | Convert to numeric; fill missing zero-tenure records with configured zero value | Blank total charges occur for new customers with no accumulated charges |
| Other numeric columns | Fill with training-set median by default | Robust to outliers and avoids dropping rows |

## Encoding Strategy

- Encoder: `onehot`
- Unknown category handling: `ignore`
- Categorical columns are detected automatically after cleaning and feature selection.

## Scaling Strategy

- Scaling enabled: `True`
- Scaler: `standard`
- Numerical columns are detected automatically after cleaning and feature selection.

## Split Strategy

- Train size: `0.7`
- Validation size: `0.15`
- Test size: `0.15`
- Stratified: `True`
- Random state: `42`

## Output Summary

- Final transformed feature count: `1174`
- Train rows: `4930`
- Validation rows: `1056`
- Test rows: `1057`

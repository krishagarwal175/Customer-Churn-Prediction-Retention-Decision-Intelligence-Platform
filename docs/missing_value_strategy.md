# Missing Value Strategy

This document records preprocessing decisions for missing values in the IBM Telco Customer Churn dataset.

| Column / Group | Why Missing Values Occur | Genuine Missing? | Handling Strategy |
|---|---|---|---|
| CustomerID | Should not be missing in valid source data | No | Validation should fail before preprocessing |
| Count | Reporting helper; should be present | No | Excluded from features by default |
| Country, State, Lat Long | Reporting/geographic fields; expected in source | No | Excluded from features by default |
| City, Zip Code, Latitude, Longitude | Location fields can be absent in future customer records | Possibly | Retain if configured as features; categorical missing filled with `Unknown`, numeric missing filled with median |
| Demographic columns | Source system may omit profile values | Yes | Preserve row and fill categorical missing with `Unknown` |
| Service columns | Source system may omit subscription flags | Yes | Preserve row and fill categorical missing with `Unknown` |
| Contract, Paperless Billing, Payment Method | Billing system should usually provide these values | Possibly | Fill categorical missing with `Unknown` for inference robustness |
| Monthly Charges | Should be present for active billing records | No for training data, possible for inference | Fill numeric missing with training median |
| Total Charges | Blank values occur for zero-tenure customers with no accumulated charges | Yes, but explainable | Convert to numeric; fill missing values where tenure is zero with `0.0`; remaining missing values use numeric strategy |
| Churn Label | Supervised target must be present for training | No | Required for fit; future inference records may omit it |
| Churn Value | Duplicate target | Not used | Removed from features |
| Churn Score | Existing churn score / potential leakage | Not used | Removed from features |
| Churn Reason | Missing for non-churned customers because no churn reason exists | Genuine structural missingness | Removed from features as post-event leakage |
| CLTV | Derived value may be absent for some future records | Possibly | Retain as configured feature; numeric missing filled with median |

Rows should not be blindly dropped during preprocessing. Dropping rows can bias model training and make future inference brittle. The pipeline therefore uses deterministic, training-set-learned imputations after preserving the raw validated dataset.


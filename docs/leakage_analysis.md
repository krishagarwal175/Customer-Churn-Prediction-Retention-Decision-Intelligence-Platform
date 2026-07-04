# Target Leakage Analysis

This document identifies source dataset columns that must not be used as model input features during churn prediction training.

| Column | Exclude from Training? | Reason |
|---|---|---|
| CustomerID | Yes | Unique identifier with no generalizable predictive meaning. Including it can let a model memorize customers rather than learn churn patterns. |
| Churn Label | Yes as feature | This is the primary target label. It may be used as `y`, but never as an input feature. |
| Churn Value | Yes | Numeric duplicate of the target outcome. Including it as an input would directly reveal the answer. |
| Churn Score | Yes | Existing IBM churn score. It behaves like a prior model output or post-processed risk indicator and would inflate performance unrealistically. |
| Churn Reason | Yes | Recorded only for customers who churned. It is post-event information unavailable before churn occurs. |
| Churn Category | Yes if present | Post-event churn classification. It is unavailable for active customers before churn and would leak the outcome. |
| Customer Status | Yes if present | Status fields commonly encode whether the customer has churned, stayed, or joined. This directly overlaps with the target. |

## Modeling Rule

The training feature matrix must exclude all identifier, target, duplicate-target, and post-event columns. These columns may still be used for reporting, customer lookup, validation, or business interpretation after predictions are generated.


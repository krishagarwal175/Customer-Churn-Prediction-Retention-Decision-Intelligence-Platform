# Data Dictionary

IBM Telco Customer Churn source dataset column definitions.

| Column | Type | Category | Description | Used for ML? | Leakage Risk | Notes |
|---|---|---|---|---|---|---|
| CustomerID | String | Identifier | Unique customer identifier. | No | Low | Use for lookup and joins only; exclude from training. |
| Count | Integer | Reporting Only | Record-count helper field, usually equal to 1. | No | Low | Useful for aggregation checks only. |
| Country | String | Geographic | Customer country. | No | Low | Usually constant in this dataset. |
| State | String | Geographic | Customer state. | No | Low | Usually constant in this dataset. |
| City | String | Geographic | Customer city. | Optional | Low | Useful for regional reporting; use carefully in ML. |
| Zip Code | Integer | Geographic | Customer ZIP code. | Optional | Low | High-cardinality location feature. |
| Lat Long | String | Geographic | Combined latitude and longitude text. | No | Low | Redundant with numeric latitude and longitude. |
| Latitude | Float | Geographic | Customer latitude coordinate. | Optional | Low | Useful for mapping and possible regional effects. |
| Longitude | Float | Geographic | Customer longitude coordinate. | Optional | Low | Useful for mapping and possible regional effects. |
| Gender | String | Customer Demographic | Customer gender. | Yes | Low | Categorical demographic feature. |
| Senior Citizen | String | Customer Demographic | Whether the customer is a senior citizen. | Yes | Low | Binary demographic feature. |
| Partner | String | Customer Demographic | Whether the customer has a partner. | Yes | Low | Household stability signal. |
| Dependents | String | Customer Demographic | Whether the customer has dependents. | Yes | Low | Household profile signal. |
| Tenure Months | Integer | Numerical Feature | Number of months the customer has stayed with the company. | Yes | Low | Core lifecycle and loyalty feature. |
| Phone Service | String | Service Information | Whether the customer has phone service. | Yes | Low | Service adoption feature. |
| Multiple Lines | String | Service Information | Whether the customer has multiple phone lines. | Yes | Low | Phone-service depth feature. |
| Internet Service | String | Service Information | Customer internet service type. | Yes | Low | Major product and churn-risk dimension. |
| Online Security | String | Service Information | Whether the customer subscribes to online security. | Yes | Low | Internet add-on and stickiness signal. |
| Online Backup | String | Service Information | Whether the customer subscribes to online backup. | Yes | Low | Internet add-on and stickiness signal. |
| Device Protection | String | Service Information | Whether the customer subscribes to device protection. | Yes | Low | Internet add-on and support signal. |
| Tech Support | String | Service Information | Whether the customer subscribes to tech support. | Yes | Low | Support coverage and retention signal. |
| Streaming TV | String | Service Information | Whether the customer subscribes to streaming TV. | Yes | Low | Entertainment bundle signal. |
| Streaming Movies | String | Service Information | Whether the customer subscribes to streaming movies. | Yes | Low | Entertainment bundle signal. |
| Contract | String | Categorical Feature | Customer contract type. | Yes | Low | Critical commercial retention feature. |
| Paperless Billing | String | Categorical Feature | Whether the customer uses paperless billing. | Yes | Low | Digital billing behavior feature. |
| Payment Method | String | Categorical Feature | Customer payment method. | Yes | Low | Billing friction and autopay signal. |
| Monthly Charges | Float | Financial | Customer's current monthly charge. | Yes | Low | Core revenue and price-sensitivity feature. |
| Total Charges | Float | Financial | Total amount charged to the customer historically. | Yes | Low | Historical value proxy. |
| Churn Label | String | Target | Human-readable churn target: Yes or No. | Target | High if used as feature | Use as the primary supervised target, not as an input feature. |
| Churn Value | Integer | Target | Numeric churn target: 1 for churn, 0 otherwise. | No | High | Duplicate target representation; exclude from features. |
| Churn Score | Integer | Potential Leakage | Existing IBM churn score for the customer. | No | High | Model-like post-processed score that would inflate performance. |
| CLTV | Integer | Derived Feature | Estimated customer lifetime value. | Optional | Medium | Use primarily for prioritization and simulation; treat carefully in ML. |
| Churn Reason | String | Potential Leakage | Reason recorded for customers who churned. | No | High | Known only after churn; exclude from training. |

"""Data dictionary metadata for the IBM Telco dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataDictionaryEntry:
    """Business metadata for one dataset column."""

    column: str
    dtype: str
    category: str
    description: str
    used_for_ml: str
    leakage_risk: str
    notes: str


DATA_DICTIONARY: tuple[DataDictionaryEntry, ...] = (
    DataDictionaryEntry(
        "CustomerID",
        "String",
        "Identifier",
        "Unique customer identifier.",
        "No",
        "Low",
        "Use for lookup and joins only; exclude from training.",
    ),
    DataDictionaryEntry(
        "Count",
        "Integer",
        "Reporting Only",
        "Record-count helper field, usually equal to 1.",
        "No",
        "Low",
        "Useful for aggregation checks only.",
    ),
    DataDictionaryEntry(
        "Country",
        "String",
        "Geographic",
        "Customer country.",
        "No",
        "Low",
        "Usually constant in this dataset.",
    ),
    DataDictionaryEntry(
        "State",
        "String",
        "Geographic",
        "Customer state.",
        "No",
        "Low",
        "Usually constant in this dataset.",
    ),
    DataDictionaryEntry(
        "City",
        "String",
        "Geographic",
        "Customer city.",
        "Optional",
        "Low",
        "Useful for regional reporting; use carefully in ML.",
    ),
    DataDictionaryEntry(
        "Zip Code",
        "Integer",
        "Geographic",
        "Customer ZIP code.",
        "Optional",
        "Low",
        "High-cardinality location feature.",
    ),
    DataDictionaryEntry(
        "Lat Long",
        "String",
        "Geographic",
        "Combined latitude and longitude text.",
        "No",
        "Low",
        "Redundant with numeric latitude and longitude.",
    ),
    DataDictionaryEntry(
        "Latitude",
        "Float",
        "Geographic",
        "Customer latitude coordinate.",
        "Optional",
        "Low",
        "Useful for mapping and possible regional effects.",
    ),
    DataDictionaryEntry(
        "Longitude",
        "Float",
        "Geographic",
        "Customer longitude coordinate.",
        "Optional",
        "Low",
        "Useful for mapping and possible regional effects.",
    ),
    DataDictionaryEntry(
        "Gender",
        "String",
        "Customer Demographic",
        "Customer gender.",
        "Yes",
        "Low",
        "Categorical demographic feature.",
    ),
    DataDictionaryEntry(
        "Senior Citizen",
        "String",
        "Customer Demographic",
        "Whether the customer is a senior citizen.",
        "Yes",
        "Low",
        "Binary demographic feature.",
    ),
    DataDictionaryEntry(
        "Partner",
        "String",
        "Customer Demographic",
        "Whether the customer has a partner.",
        "Yes",
        "Low",
        "Household stability signal.",
    ),
    DataDictionaryEntry(
        "Dependents",
        "String",
        "Customer Demographic",
        "Whether the customer has dependents.",
        "Yes",
        "Low",
        "Household profile signal.",
    ),
    DataDictionaryEntry(
        "Tenure Months",
        "Integer",
        "Numerical Feature",
        "Number of months the customer has stayed with the company.",
        "Yes",
        "Low",
        "Core lifecycle and loyalty feature.",
    ),
    DataDictionaryEntry(
        "Phone Service",
        "String",
        "Service Information",
        "Whether the customer has phone service.",
        "Yes",
        "Low",
        "Service adoption feature.",
    ),
    DataDictionaryEntry(
        "Multiple Lines",
        "String",
        "Service Information",
        "Whether the customer has multiple phone lines.",
        "Yes",
        "Low",
        "Phone-service depth feature.",
    ),
    DataDictionaryEntry(
        "Internet Service",
        "String",
        "Service Information",
        "Customer internet service type.",
        "Yes",
        "Low",
        "Major product and churn-risk dimension.",
    ),
    DataDictionaryEntry(
        "Online Security",
        "String",
        "Service Information",
        "Whether the customer subscribes to online security.",
        "Yes",
        "Low",
        "Internet add-on and stickiness signal.",
    ),
    DataDictionaryEntry(
        "Online Backup",
        "String",
        "Service Information",
        "Whether the customer subscribes to online backup.",
        "Yes",
        "Low",
        "Internet add-on and stickiness signal.",
    ),
    DataDictionaryEntry(
        "Device Protection",
        "String",
        "Service Information",
        "Whether the customer subscribes to device protection.",
        "Yes",
        "Low",
        "Internet add-on and support signal.",
    ),
    DataDictionaryEntry(
        "Tech Support",
        "String",
        "Service Information",
        "Whether the customer subscribes to tech support.",
        "Yes",
        "Low",
        "Support coverage and retention signal.",
    ),
    DataDictionaryEntry(
        "Streaming TV",
        "String",
        "Service Information",
        "Whether the customer subscribes to streaming TV.",
        "Yes",
        "Low",
        "Entertainment bundle signal.",
    ),
    DataDictionaryEntry(
        "Streaming Movies",
        "String",
        "Service Information",
        "Whether the customer subscribes to streaming movies.",
        "Yes",
        "Low",
        "Entertainment bundle signal.",
    ),
    DataDictionaryEntry(
        "Contract",
        "String",
        "Categorical Feature",
        "Customer contract type.",
        "Yes",
        "Low",
        "Critical commercial retention feature.",
    ),
    DataDictionaryEntry(
        "Paperless Billing",
        "String",
        "Categorical Feature",
        "Whether the customer uses paperless billing.",
        "Yes",
        "Low",
        "Digital billing behavior feature.",
    ),
    DataDictionaryEntry(
        "Payment Method",
        "String",
        "Categorical Feature",
        "Customer payment method.",
        "Yes",
        "Low",
        "Billing friction and autopay signal.",
    ),
    DataDictionaryEntry(
        "Monthly Charges",
        "Float",
        "Financial",
        "Customer's current monthly charge.",
        "Yes",
        "Low",
        "Core revenue and price-sensitivity feature.",
    ),
    DataDictionaryEntry(
        "Total Charges",
        "Float",
        "Financial",
        "Total amount charged to the customer historically.",
        "Yes",
        "Low",
        "Historical value proxy.",
    ),
    DataDictionaryEntry(
        "Churn Label",
        "String",
        "Target",
        "Human-readable churn target: Yes or No.",
        "Target",
        "High if used as feature",
        "Use as the primary supervised target, not as an input feature.",
    ),
    DataDictionaryEntry(
        "Churn Value",
        "Integer",
        "Target",
        "Numeric churn target: 1 for churn, 0 otherwise.",
        "No",
        "High",
        "Duplicate target representation; exclude from features.",
    ),
    DataDictionaryEntry(
        "Churn Score",
        "Integer",
        "Potential Leakage",
        "Existing IBM churn score for the customer.",
        "No",
        "High",
        "Model-like post-processed score that would inflate performance.",
    ),
    DataDictionaryEntry(
        "CLTV",
        "Integer",
        "Derived Feature",
        "Estimated customer lifetime value.",
        "Optional",
        "Medium",
        "Use primarily for prioritization and simulation; treat carefully in ML.",
    ),
    DataDictionaryEntry(
        "Churn Reason",
        "String",
        "Potential Leakage",
        "Reason recorded for customers who churned.",
        "No",
        "High",
        "Known only after churn; exclude from training.",
    ),
)


def render_data_dictionary_markdown(
    entries: tuple[DataDictionaryEntry, ...] = DATA_DICTIONARY,
) -> str:
    """Render the data dictionary as a Markdown table."""

    lines = [
        "# Data Dictionary",
        "",
        "IBM Telco Customer Churn source dataset column definitions.",
        "",
        "| Column | Type | Category | Description | Used for ML? | Leakage Risk | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            "| "
            f"{entry.column} | {entry.dtype} | {entry.category} | "
            f"{entry.description} | {entry.used_for_ml} | "
            f"{entry.leakage_risk} | {entry.notes} |"
        )
    return "\n".join(lines) + "\n"


def save_data_dictionary(output_path: str | Path) -> None:
    """Save the data dictionary Markdown document."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_data_dictionary_markdown(), encoding="utf-8")
    LOGGER.info("Data dictionary saved to %s", path)

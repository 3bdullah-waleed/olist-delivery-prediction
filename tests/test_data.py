import pandas as pd
from src.features import FEATURE_COLUMNS

# Columns that must NEVER appear in the model's feature list —
# they contain information only available after the prediction moment
LEAKAGE_COLUMNS = [
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_score",
    "review_comment_message",
    "order_approved_at",
    "order_delivered_carrier_date",
]


def test_no_leakage_columns_in_features():
    """FEATURE_COLUMNS must never contain post-delivery information."""
    for col in LEAKAGE_COLUMNS:
        assert col not in FEATURE_COLUMNS, f"Leakage column '{col}' found in FEATURE_COLUMNS!"


def test_is_late_label_is_binary():
    """is_late must only contain 0 or 1."""
    train = pd.read_csv("data/train_features.csv")
    unique_values = set(train["is_late"].unique())
    assert unique_values.issubset({0, 1}), f"is_late has unexpected values: {unique_values}"


def test_no_nulls_in_feature_columns():
    """FEATURE_COLUMNS should have no missing values after feature engineering."""
    train = pd.read_csv("data/train_features.csv")
    for col in FEATURE_COLUMNS:
        assert train[col].isnull().sum() == 0, f"Column '{col}' has missing values"
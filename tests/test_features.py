import pandas as pd
from src.features import add_features, FEATURE_COLUMNS


def test_add_features_creates_expected_columns():
    """add_features should create purchase_month, same_state, and log_total_price."""
    sample = pd.DataFrame({
        "order_purchase_timestamp": ["2018-01-15 10:00:00"],
        "num_items": [2],
        "total_price": [100.0],
        "total_freight": [15.0],
        "num_payments": [1],
        "total_payment_value": [115.0],
        "customer_state": ["SP"],
        "seller_state": ["SP"],
    })

    result = add_features(sample)

    assert "purchase_month" in result.columns
    assert "same_state" in result.columns
    assert "log_total_price" in result.columns


def test_same_state_is_zero_when_states_differ():
    
    sample = pd.DataFrame({
        "order_purchase_timestamp": ["2018-01-15 10:00:00"],
        "num_items": [1],
        "total_price": [50.0],
        "total_freight": [10.0],
        "num_payments": [1],
        "total_payment_value": [60.0],
        "customer_state": ["SP"],
        "seller_state": ["RJ"],
    })

def test_same_state_is_one_when_states_match():
    """same_state should be 1 when customer_state equals seller_state."""
    sample = pd.DataFrame({
        "order_purchase_timestamp": ["2018-01-15 10:00:00"],
        "num_items": [1],
        "total_price": [50.0],
        "total_freight": [10.0],
        "num_payments": [1],
        "total_payment_value": [60.0],
        "customer_state": ["SP"],
        "seller_state": ["SP"],
    })

    result = add_features(sample)

    assert result["same_state"].iloc[0] == 1
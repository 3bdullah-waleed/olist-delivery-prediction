import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering used in training to any DataFrame,
    whether it has thousands of rows (training) or a single row (API inference).
    """
    try:
        df = df.copy()

        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

        df["num_items"] = df["num_items"].fillna(0)
        df["total_price"] = df["total_price"].fillna(0)
        df["total_freight"] = df["total_freight"].fillna(0)
        df["num_payments"] = df["num_payments"].fillna(0)
        df["total_payment_value"] = df["total_payment_value"].fillna(0)

        df["log_total_price"] = np.log1p(df["total_price"])

        df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

        logger.info(f"Features added successfully for {len(df)} row(s)")
        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


FEATURE_COLUMNS = [
    "num_items", "total_freight", "log_total_price",
    "num_payments", "total_payment_value",
    "purchase_month", "purchase_dayofweek",
    "same_state", "num_unique_sellers"
]
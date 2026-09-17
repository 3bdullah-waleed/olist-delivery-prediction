import pandas as pd
import great_expectations as gx
from src.logger import get_logger

logger = get_logger(__name__)

VALID_STATES = [
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 
    'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 
    'SE', 'SP', 'TO'
]


def validate_order(df: pd.DataFrame) -> bool:
    """
    Validate an incoming order (or batch of orders) before feature engineering.
    Returns True if valid, raises ValueError if any check fails.
    """
    context = gx.get_context()
    data_source = context.data_sources.add_pandas("temp_source")
    data_asset = data_source.add_dataframe_asset(name="temp_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("temp_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="order_validation_suite")

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="customer_state", value_set=VALID_STATES
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="seller_state", value_set=VALID_STATES
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_price", min_value=0, max_value=None
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="num_items", value_set=None, mostly=1.0
        ) if False else gx.expectations.ExpectColumnValuesToBeBetween(
            column="num_items", min_value=1, max_value=None
        )
    )

    results = batch.validate(suite)

    if not results.success:
        logger.error(f"Validation failed: {results}")
        raise ValueError("Order validation failed. Check input data.")

    logger.info(f"Validation passed for {len(df)} row(s)")
    return True
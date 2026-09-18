import pandas as pd
from src.model import load_model
from src.features import add_features, FEATURE_COLUMNS

def get_sample_row():
    """Helper: build a single valid order row for testing."""
    train = pd.read_csv("data/train.csv")
    return add_features(train.head(1))


def test_model_loads_successfully():
    """The model should load without errors and not be None."""
    model = load_model()
    assert model is not None


def test_model_predict_proba_shape_is_correct():
    """predict_proba should return one row per input, with 2 columns (class 0 and class 1)."""
    model = load_model()
    row = get_sample_row()

    proba = model.predict_proba(row[FEATURE_COLUMNS])

    assert proba.shape == (1, 2)


def test_model_probability_is_within_valid_range():
    """Predicted probabilities must be valid probabilities: between 0 and 1."""
    model = load_model()
    row = get_sample_row()

    proba = model.predict_proba(row[FEATURE_COLUMNS])[:, 1]

    assert (proba >= 0).all() and (proba <= 1).all()
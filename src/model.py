import joblib
import os
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "logistic_regression_v1.pkl")


def load_model():
    """Load the trained model from a local file (exported from MLflow)."""
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
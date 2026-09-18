import mlflow
from src.logger import get_logger

logger = get_logger(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "olist_late_delivery_model"
MODEL_ALIAS = "champion"


def load_model():
    """Load the current 'champion' model from MLflow Model Registry."""
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded successfully from MLflow registry: {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow registry: {e}")
        raise
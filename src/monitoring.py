import csv
import os
from datetime import datetime
from src.logger import get_logger

logger = get_logger(__name__)

PREDICTIONS_LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.csv")


def log_prediction(order_id: str, prediction: int, probability: float, latency_ms: float):
    """Append a single prediction record to the predictions log for later evaluation."""
    file_exists = os.path.isfile(PREDICTIONS_LOG)
    try:
        with open(PREDICTIONS_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "order_id", "prediction", "probability", "latency_ms"])
            writer.writerow([datetime.utcnow().isoformat(), order_id, prediction, probability, latency_ms])
        logger.info(f"Prediction logged for order_id={order_id}")
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.model import load_model
from src.features import add_features, FEATURE_COLUMNS
from src.validation import validate_order
from src.logger import get_logger
import time
from src.monitoring import log_prediction

logger = get_logger(__name__)

app = FastAPI(title="Olist Late Delivery Prediction API")

model = load_model()


class OrderInput(BaseModel):
    order_id: str
    order_purchase_timestamp: str
    num_items: float
    total_price: float
    total_freight: float
    num_payments: float
    total_payment_value: float
    customer_state: str
    seller_state: str
    num_unique_sellers: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: OrderInput):
    start_time = time.time()
    try:
        df = pd.DataFrame([order.model_dump()])

        validate_order(df)
        df = add_features(df)

        prediction = int(model.predict(df[FEATURE_COLUMNS])[0])
        probability = float(model.predict_proba(df[FEATURE_COLUMNS])[:, 1][0])

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Prediction made: {prediction}, probability: {probability:.4f}, latency: {latency_ms:.2f}ms")
        log_prediction(order.order_id, prediction, probability, latency_ms)

        return {
            "prediction": prediction,
            "probability": probability,
            "is_late": bool(prediction)
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
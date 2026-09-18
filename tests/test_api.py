from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    """The /health endpoint should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_valid_order_returns_200():
    """A valid order should return a prediction, probability, and is_late field."""
    valid_order = {
        "order_purchase_timestamp": "2018-01-15 10:00:00",
        "num_items": 2,
        "total_price": 150.0,
        "total_freight": 20.0,
        "num_payments": 1,
        "total_payment_value": 170.0,
        "customer_state": "SP",
        "seller_state": "RJ",
        "num_unique_sellers": 1
    }

    response = client.post("/predict", json=valid_order)

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert "is_late" in body
    assert 0 <= body["probability"] <= 1


def test_predict_with_invalid_state_returns_400():
    """An order with an invalid customer_state should be rejected."""
    invalid_order = {
        "order_purchase_timestamp": "2018-01-15 10:00:00",
        "num_items": 2,
        "total_price": 150.0,
        "total_freight": 20.0,
        "num_payments": 1,
        "total_payment_value": 170.0,
        "customer_state": "XYZ",
        "seller_state": "RJ",
        "num_unique_sellers": 1
    }

    response = client.post("/predict", json=invalid_order)

    assert response.status_code == 400
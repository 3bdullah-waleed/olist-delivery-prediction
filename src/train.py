import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from src.features import FEATURE_COLUMNS
from src.logger import get_logger

import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

logger = get_logger(__name__)

mlflow.set_experiment("olist_late_delivery_prediction_v2")


def train_model():
    train = pd.read_csv("data/train_features.csv")
    val = pd.read_csv("data/val_features.csv")

    X_train, y_train = train[FEATURE_COLUMNS], train["is_late"]
    X_val, y_val = val[FEATURE_COLUMNS], val["is_late"]

    params = {"class_weight": "balanced", "max_iter": 1000}

    with mlflow.start_run():
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_preds = model.predict(X_val)
        auc = roc_auc_score(y_val, val_proba)

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("auc", auc)

        # Log the model itself
        mlflow.sklearn.log_model(model, "model", registered_model_name="olist_late_delivery_model")

        logger.info(f"Training run completed. AUC: {auc:.4f}")
        print(classification_report(y_val, val_preds))
        print("AUC:", auc)

    return model


if __name__ == "__main__":
    train_model()
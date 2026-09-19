import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
    name="olist_late_delivery_model",
    alias="champion",
    version=1
)
print("Alias set successfully")
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient()
mv = client.get_latest_versions("olist_late_delivery_model")[0]
print("Version:", mv.version)
print("Source:", mv.source)
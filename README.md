## Known Limitation: MLflow Artifact Serving in Docker

MLflow tracking and model registry are fully functional (params, metrics, and 
model versions are logged and visible in the MLflow UI, both locally and via 
Docker Compose). However, loading a registered model's artifact directly inside 
a container consistently failed with `MlflowException: No such artifact: ''`, 
traced to `local_artifact_repo.py` attempting to resolve the artifact as a local 
path rather than through the HTTP artifact proxy — even after training was 
run entirely within the same Docker network as the MLflow server.

**Decision:** The API loads the model from a local `.pkl` file (exported via 
joblib) rather than from the MLflow registry at runtime. MLflow remains the 
system of record for experiment tracking and model versioning during 
development; the `.pkl` file represents a manually "promoted" snapshot of the 
champion model for deployment.

**Future improvement:** Investigate MLflow's artifact proxy configuration 
(`--serve-artifacts` flag behavior across versions) or switch to an S3/MinIO-
backed artifact store, which is the standard production pattern for this exact 
problem.

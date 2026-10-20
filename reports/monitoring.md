# Monitoring Strategy

## 1. Application Health

The FastAPI application provides a `/health` endpoint that reports whether the service is running and the model is loaded. Kubernetes readiness and liveness probes are configured to check this endpoint.

## 2. Application Logs

Uvicorn logs are used to monitor API requests and identify HTTP errors. For the Docker deployment, logs can be inspected with:

`docker logs heart-disease-api-container`

## 3. Model Monitoring

The project records model evaluation metrics, including accuracy, precision, recall, F1-score, and ROC-AUC. These metrics provide a baseline for future model comparisons.

For a production system, prediction distributions, input data quality, model drift, latency, and error rates should be monitored over time.

## 4. Limitations

This project implements basic health checks and application logging, not a complete production monitoring platform. Automated drift detection, alerting, and continuous performance evaluation remain future improvements.

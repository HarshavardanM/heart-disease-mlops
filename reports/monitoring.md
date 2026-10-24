# Monitoring and Observability

## 1. Application Health

The FastAPI application exposes a `/health` endpoint that reports whether the application is running and the model is loaded. Kubernetes readiness and liveness probes use this endpoint to check application health.

## 2. Application Logging

Uvicorn provides application and HTTP request logs. Logs from a running Docker container can be inspected using:

```powershell
docker logs heart-disease-api-container
```

For Kubernetes, application logs can be viewed using `kubectl logs`.

## 3. Prometheus Metrics

The API is instrumented to expose Prometheus metrics through the `/metrics` endpoint. A Kubernetes Service exposes the application, and a ServiceMonitor in the `monitoring` namespace configures Prometheus to scrape the service's metrics endpoint at 15-second intervals.

The monitoring configuration was verified using the Prometheus query:

```promql
up{service="heart-disease-api-service"}
```

A value of `1` indicates that Prometheus successfully scraped the API target; a value of `0` indicates a scrape failure for that target.

The `http_requests_total` metric records HTTP request counts with labels such as endpoint, HTTP method, and status category. PromQL queries can calculate request rates and inspect HTTP 5xx responses.

## 4. Grafana Dashboard

A Grafana dashboard visualizes API availability, request rate by endpoint, cumulative requests by endpoint, and HTTP server errors. The dashboard uses Prometheus as its data source.

A zero value in the HTTP server errors panel indicates that no matching HTTP 5xx responses were observed during the selected query window. It does not guarantee that every possible application or model error has been detected.

## 5. Model Evaluation

The project records model evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC during model development. These are evaluation results, not live production model-performance metrics.

## 6. Limitations and Future Work

This is a local educational deployment, not a production clinical system. Prometheus and Grafana monitor service availability and HTTP activity; they do not establish clinical safety or detect every model failure.

Future improvements could include latency percentiles, structured logging, alert rules, input-data validation metrics, drift detection, and monitoring of model performance when reliable ground-truth labels become available.

# Monitoring and Observability

## 1. API Health and Request Metrics

The Heart Disease prediction API is implemented using FastAPI. It exposes a `/health` endpoint to report application and model availability, and a `/metrics` endpoint containing Prometheus-compatible metrics. The API is instrumented using `prometheus-fastapi-instrumentator` to collect HTTP request metrics.

## 2. Kubernetes Deployment

The API runs in a local Kubernetes cluster. A Kubernetes Deployment manages the API pod, and a NodePort Service exposes the application within the local development environment. Readiness and liveness probes use the `/health` endpoint.

The deployment uses the Docker image `heart-disease-api:v2`, which includes the Prometheus instrumentation.

## 3. Prometheus Integration

Prometheus is installed using the `kube-prometheus-stack` Helm chart. A ServiceMonitor in the `monitoring` namespace discovers the API Service and scrapes its `/metrics` endpoint every 15 seconds.

The API target was verified using the Prometheus query:

```promql
up{service="heart-disease-api-service"}
```

The current API target reports a value of `1`, indicating that Prometheus can successfully scrape its metrics endpoint. Historical or removed pod instances may appear in time-series graphs for part of the selected time range.

## 4. Grafana Dashboard

Grafana visualizes the API's availability, request rate by endpoint, total requests by endpoint, and HTTP server errors. The availability panel uses the following PromQL expression:

```promql
max(up{service="heart-disease-api-service"}) or vector(0)
```

The request panels help demonstrate API activity, while the HTTP server error panel tracks server-side HTTP errors. A value of zero for server errors is expected when no HTTP 5xx responses have occurred during the selected period.

## 5. Testing and Operational Verification

The API test suite contains four tests covering the root endpoint, health endpoint, prediction endpoint, and rejection of an invalid age. All four tests passed locally. Kubernetes pod and Service status, API health, and Prometheus target health were also checked.

## 6. Limitations and Future Improvements

This is a local educational deployment rather than a production clinical system. Prometheus and Grafana provide basic operational visibility, but they do not establish model accuracy, fairness, or clinical safety. Further improvements could include latency percentile dashboards, alert rules, structured logging, input-data drift monitoring, and model-performance monitoring when reliable outcome labels become available.

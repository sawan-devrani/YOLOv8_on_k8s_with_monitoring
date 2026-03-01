# YOLOv8 Inference Service — Kubernetes Deployment with Observability

Production-style object detection inference API deployed on Kubernetes with full 
metrics observability. Built to explore real-world MLOps patterns — containerized 
model serving, horizontal scaling, and monitoring — entirely on local infrastructure.

---


## Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| Inference | YOLOv8n (Ultralytics) | Object detection, CPU-optimized |
| API | FastAPI + Uvicorn | Async inference server |
| Instrumentation | prometheus-fastapi-instrumentator | HTTP + custom metrics |
| Orchestration | Kubernetes (Docker Desktop) | Deployment, scaling, health checks |
| Metrics | Prometheus | Time-series collection and storage |
| Visualization | Grafana | Dashboard and query layer |
| Packaging | Helm | Monitoring stack deployment |

---

## Key Design Decisions

**CPU-only inference** — The deployment targets environments without GPU access.
YOLOv8n (nano) was chosen for its balance of accuracy and CPU inference speed,
averaging ~250ms per image after model warmup.

**2-replica deployment** — Provides basic availability and demonstrates K8s
load balancing across pods. Resource limits are set to 2 CPU cores and 768Mi
memory per pod.

**Metrics-first instrumentation** — Custom Prometheus metrics are defined at
the application layer rather than inferred from infrastructure, giving precise
visibility into ML-specific behavior: per-class detection rates, inference
duration distribution, and throughput.

---

## Project Structure

```
.
├── app/
│   ├── main.py                   # FastAPI inference server with metrics
│   ├── requirements.txt
│   └── yolov8n.pt                # YOLOv8 nano weights
├── Dockerfile                    # linux/amd64, CPU-only torch
├── k8s/
│   ├── deployment.yaml           # 2 replicas, probes, resource limits
│   ├── service.yaml              # NodePort :30800
│   └── networkpolicy.yaml        # default-deny, port 8000 + DNS egress
└── monitoring/
    ├── prometheus-values.yaml    # Helm overrides
    └── grafana-values.yaml       # Datasource pre-configured
```

---

## Deployment

**Prerequisites:** Docker Desktop (Kubernetes enabled), Helm 3

```bash
# Build image for linux/amd64 (Docker Desktop K8s node architecture)
docker buildx build --platform linux/amd64 \
  -t <image_name>:v1 --load .

# Deploy inference service
kubectl config use-context docker-desktop
kubectl apply -f k8s/

# Deploy monitoring stack
kubectl create namespace monitoring
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/prometheus \
  --namespace monitoring -f monitoring/prometheus-values.yaml

helm install grafana grafana/grafana \
  --namespace monitoring -f monitoring/grafana-values.yaml

# Register YOLO pods as Prometheus scrape targets
kubectl get pods -l app=yolo-inference -o wide -n default
kubectl edit configmap prometheus-server -n monitoring
kubectl rollout restart deployment prometheus-server -n monitoring
```

---

## Observability

**Exposed metrics:**

| Metric | Type | Description |
|---|---|---|
| `yolo_inference_duration_seconds` | Histogram | Per-request model inference time |
| `yolo_detections_per_image` | Histogram | Object count distribution per image |
| `yolo_images_processed_total` | Counter | Cumulative inference requests |
| `yolo_detected_class_total` | Counter | Per-class detection frequency |
| `http_request_duration_seconds` | Histogram | Full request latency by endpoint |

**Key Grafana queries:**

```promql
# Inference latency percentiles
histogram_quantile(0.95, rate(yolo_inference_duration_seconds_bucket[5m]))

# Request throughput
rate(http_requests_total{handler="/predict"}[1m])

# Average detections per image
rate(yolo_detections_per_image_sum[5m]) / rate(yolo_detections_per_image_count[5m])

# Detection distribution by class
topk(10, yolo_detected_class_total)
```

---

## Service Endpoints

| Service | URL | 
|---|---|
| Inference API | `http://localhost:30800` |
| Metrics | `http://localhost:30800/metrics` |
| Prometheus | `http://localhost:30900` |
| Grafana | `http://localhost:32000` |

---

## Known Limitations

- **Static scrape targets** — Prometheus is configured with pod IPs directly.
  Pod IP changes on restart require a manual ConfigMap update. Resolved in
  production with a Prometheus Operator `ServiceMonitor`.

- **No persistent storage** — Prometheus and Grafana use ephemeral storage.
  Dashboards and metrics are lost on pod restart. Acceptable for local
  development; requires `PersistentVolume` configuration for production.

- **CPU inference only** — Docker Desktop on macOS does not support GPU
  passthrough to the Kubernetes VM. Cloud deployment with Nvidia GPU nodes
  would bring inference latency from ~250ms to ~15ms.

---
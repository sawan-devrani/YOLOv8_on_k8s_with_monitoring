from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from PIL import Image
import io, time

app = FastAPI(title="YOLOv8 Inference API")
model = YOLO("yolov8n.pt")

# ---- Custom YOLO-specific metrics ----
INFERENCE_TIME = Histogram(
    "yolo_inference_duration_seconds",
    "Time spent running YOLO inference",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)
DETECTION_COUNT = Histogram(
    "yolo_detections_per_image",
    "Number of objects detected per image",
    buckets=[0, 1, 2, 5, 10, 20, 50]
)
TOTAL_IMAGES = Counter(
    "yolo_images_processed_total",
    "Total number of images processed"
)
CLASS_COUNTER = Counter(
    "yolo_detected_class_total",
    "Total detections per class",
    ["class_name"]
)

# ---- Auto-instrument all HTTP endpoints ----
Instrumentator().instrument(app).expose(app)

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    start = time.time()
    results = model(image, device="cpu", verbose=False)
    elapsed = time.time() - start

    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            detections.append({
                "class": cls_name,
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
            })
            CLASS_COUNTER.labels(class_name=cls_name).inc()

    # Record metrics
    INFERENCE_TIME.observe(elapsed)
    DETECTION_COUNT.observe(len(detections))
    TOTAL_IMAGES.inc()

    return JSONResponse({
        "filename": file.filename,
        "detections": detections,
        "count": len(detections),
        "inference_ms": round(elapsed * 1000, 2)
    })

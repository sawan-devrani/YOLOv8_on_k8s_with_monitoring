import requests
from pathlib import Path

BASE_URL = "http://localhost:30800"

def test_health():
    r = requests.get(f"{BASE_URL}/healthz")
    assert r.status_code == 200
    print("Health check:", r.json())

def test_predict_url_image():
    # Download a sample image (COCO bus image)
    import urllib.request
    img_url = "https://ultralytics.com/images/bus.jpg"
    urllib.request.urlretrieve(img_url, "/tmp/bus.jpg")

    with open("/tmp/bus.jpg", "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("bus.jpg", f, "image/jpeg")}
        )
    
    assert r.status_code == 200
    data = r.json()
    print(f"Detections found: {data['count']}")
    for d in data["detections"]:
        print(f"   → {d['class']} ({d['confidence']*100:.1f}%) at {d['bbox']}")

if __name__ == "__main__":
    test_health()
    test_predict_url_image()

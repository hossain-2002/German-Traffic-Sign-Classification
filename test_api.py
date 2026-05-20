"""Quick script to test the prediction API with multiple images."""
import requests

API_URL = "http://localhost:8000/predict"

test_images = [
    "data/Test/00000.png",
    "data/Test/00001.png",
    "data/Test/00002.png",
    "data/Test/00003.png",
    "data/Test/00004.png",
]

print(f"{'Image':<25} {'Prediction':<40} {'Confidence'}")
print("-" * 80)

for img_path in test_images:
    with open(img_path, "rb") as f:
        r = requests.post(API_URL, files={"file": f})
    result = r.json()
    print(f"{img_path:<25} {result['predicted_class_name']:<40} {result['confidence']:.4f}")

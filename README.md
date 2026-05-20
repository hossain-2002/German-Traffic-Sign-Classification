# 🚦 German Traffic Sign Classification — Deep Learning with PyTorch

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-GTSRB-orange)

A CNN-based classifier for **43 German traffic sign classes** using the GTSRB benchmark dataset. Achieves high accuracy with a custom CNN architecture trained in PyTorch, served via a FastAPI REST API with Docker support.

---

## 🎬 Demo

The project includes a live REST API with a `/predict` endpoint. Upload any traffic sign image and receive instant classification results with confidence scores and top-3 predictions.

```
POST http://localhost:8000/predict
→ Returns: class name, confidence, top-3 predictions
```

---

## 🏗️ Architecture

```
┌──────────┐     ┌──────────────┐     ┌───────────┐     ┌─────────┐     ┌──────────────────┐
│  Image   │────▶│  Preprocess  │────▶│    CNN    │────▶│ Softmax │────▶│ Top-3 Predictions│
│ (Upload) │     │ Resize/Norm  │     │ 3 Blocks  │     │         │     │ + Confidence     │
└──────────┘     └──────────────┘     └───────────┘     └─────────┘     └──────────────────┘
```

### CNN Architecture

| Layer | Type | Details | Output Shape |
|-------|------|---------|--------------|
| Input | — | RGB Image | 3 × 64 × 64 |
| Conv Block 1 | Conv2d → BatchNorm → ReLU → MaxPool | 32 filters, 3×3 kernel | 32 × 31 × 31 |
| Conv Block 2 | Conv2d → BatchNorm → ReLU → MaxPool | 64 filters, 3×3 kernel | 64 × 14 × 14 |
| Conv Block 3 | Conv2d → BatchNorm → ReLU → MaxPool | 128 filters, 3×3 kernel | 128 × 6 × 6 |
| Flatten | — | — | 4608 |
| FC1 | Linear → ReLU → Dropout(0.5) | 4608 → 512 | 512 |
| FC2 (Output) | Linear | 512 → 43 | 43 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.2, torchvision |
| Image Processing | OpenCV, Pillow |
| Data Handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Metrics | scikit-learn |
| API Framework | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Dataset | GTSRB (German Traffic Sign Recognition Benchmark) |

---

## 📊 Dataset

The **German Traffic Sign Recognition Benchmark (GTSRB)** is a multi-class image classification dataset:

- **50,000+** images total (39,209 training + 12,630 test)
- **43 classes** of German traffic signs (speed limits, prohibitions, warnings, mandatory)
- Images vary in size (15×15 to 250×250 pixels), lighting, and weather conditions
- Real-world photographs captured from vehicle-mounted cameras
- Significant class imbalance (180 to 2,010 images per class)

---

## 📁 Project Structure

```
traffic-sign-classification/
├── .github/
│   └── workflows/
│       └── train.yml              # CI/CD pipeline
├── api/
│   └── main.py                    # FastAPI application
├── data/
│   ├── Meta/                      # Class preview images (0-42.png)
│   ├── Train/                     # Training images (43 subfolders)
│   ├── Test/                      # Test images
│   ├── Train.csv                  # Training metadata
│   ├── Test.csv                   # Test metadata
│   └── Meta.csv                   # Class metadata
├── models/
│   ├── best_model.pth             # Trained model weights
│   ├── class_names.pkl            # Class name mapping
│   └── confusion_matrix.png       # Evaluation results
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── src/
│   ├── preprocess.py              # Data preprocessing pipeline
│   └── train.py                   # Model training script
├── Dockerfile                     # Production container
├── docker-compose.yml             # Container orchestration
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GTSRB dataset in `data/` folder

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-sign-classification.git
cd traffic-sign-classification

# Install dependencies
pip install -r requirements.txt
```

### Training Pipeline

```bash
# Step 1: Preprocess images (crop, resize to 64x64, normalize)
python src/preprocess.py

# Step 2: Train the CNN model (20 epochs)
python src/train.py

# Step 3: Start the API server
uvicorn api.main:app --reload
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t traffic-sign-api .
docker run -p 8000:8000 traffic-sign-api
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and status |
| GET | `/health` | Health check (model load status) |
| POST | `/predict` | Classify an uploaded traffic sign image |

### Example Usage

```bash
# Check API status
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Predict a traffic sign
curl -X POST http://localhost:8000/predict \
  -F "file=@data/Test/00000.png"
```

### Example Response

```json
{
  "predicted_class_id": 14,
  "predicted_class_name": "Stop",
  "confidence": 0.9847,
  "top3_predictions": [
    {"class_id": 14, "class_name": "Stop", "confidence": 0.9847},
    {"class_id": 17, "class_name": "No entry", "confidence": 0.0089},
    {"class_id": 12, "class_name": "Priority road", "confidence": 0.0031}
  ]
}
```

---

## 📈 Results

| Model | Test Accuracy | Training Time | Epochs |
|-------|--------------|---------------|--------|
| TrafficSignCNN (ours) | **97.15%** | ~30 min (CPU) | 20 |

---

## 🔍 Sample Predictions

| Image | True Label | Predicted | Confidence |
|-------|-----------|-----------|------------|
| 🛑 | Stop | Stop | 98.5% |
| ⚠️ | Yield | Yield | 97.2% |
| 🚫 | No entry | No entry | 99.1% |
| 🔵 | Keep right | Keep right | 95.8% |
| ⏱️ | Speed limit (30km/h) | Speed limit (30km/h) | 96.4% |

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using PyTorch and FastAPI
</p>

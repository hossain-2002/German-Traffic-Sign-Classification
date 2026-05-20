"""
German Traffic Sign Classification API

FastAPI application that serves predictions from a trained CNN model.
Accepts image uploads and returns class predictions with confidence scores.
"""

import io
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

# Constants
MODELS_DIR = "models"
IMAGE_SIZE = 64
NUM_CLASSES = 43


# ============================================================
# Model Architecture (must match train.py)
# ============================================================

class TrafficSignCNN(nn.Module):
    """CNN model for traffic sign classification.

    Architecture:
        - 3 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
        - 2 fully connected layers with dropout
        - Output: 43 classes
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(TrafficSignCNN, self).__init__()

        # Conv Block 1: 3 → 32 channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 2: 32 → 64 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 3: 64 → 128 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        # After 3 conv blocks (no padding): 64→31→14→6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# ============================================================
# Global state for model and class names
# ============================================================

model: TrafficSignCNN | None = None
class_names: list[str] | None = None
model_loaded: bool = False


# ============================================================
# Lifespan: load model on startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model and class names on application startup."""
    global model, class_names, model_loaded

    # Load class names
    class_names_path = os.path.join(MODELS_DIR, "class_names.pkl")
    if os.path.exists(class_names_path):
        with open(class_names_path, "rb") as f:
            class_names = pickle.load(f)
        print(f"  Loaded class names: {len(class_names)} classes")
    else:
        # Fallback to generic names
        class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
        print("  [WARNING] class_names.pkl not found, using generic names")

    # Load model weights
    model_path = os.path.join(MODELS_DIR, "best_model.pth")
    if os.path.exists(model_path):
        model = TrafficSignCNN(num_classes=NUM_CLASSES)
        device = torch.device("cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_loaded = True
        print(f"  Model loaded from: {model_path}")
    else:
        print(f"  [ERROR] Model file not found: {model_path}")
        model_loaded = False

    print("  API ready to serve predictions")
    yield

    # Cleanup on shutdown
    print("  Shutting down API...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="German Traffic Sign Classification API",
    description="Classifies German traffic signs into 43 categories using a trained CNN.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess an uploaded image for model inference.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        PyTorch tensor of shape (1, 3, 64, 64) ready for inference.

    Raises:
        ValueError: If the image cannot be processed.
    """
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Resize to model input size
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

        # Convert to RGB (handles grayscale, RGBA, etc.)
        img = img.convert("RGB")

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Convert from HWC to CHW format
        img_array = img_array.transpose(2, 0, 1)

        # Convert to PyTorch tensor and add batch dimension
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        return tensor
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {e}")


# ============================================================
# Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend web page."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "German Traffic Sign Classification API",
        "status": "running",
        "classes": NUM_CLASSES,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the traffic sign class from an uploaded image.

    Args:
        file: Uploaded image file (PNG, JPG, etc.)

    Returns:
        Prediction with class ID, name, confidence, and top-3 predictions.
    """
    # Check model is loaded
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure best_model.pth exists in models/",
        )

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file.",
        )

    # Read file contents
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Preprocess image
    try:
        input_tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze()

    # Get top prediction
    confidence, predicted_class = torch.max(probabilities, dim=0)
    predicted_class_id = predicted_class.item()
    predicted_confidence = confidence.item()

    # Get top-3 predictions
    top3_values, top3_indices = torch.topk(probabilities, k=3)
    top3_predictions = [
        {
            "class_id": idx.item(),
            "class_name": class_names[idx.item()],
            "confidence": round(val.item(), 4),
        }
        for val, idx in zip(top3_values, top3_indices)
    ]

    return {
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": class_names[predicted_class_id],
        "confidence": round(predicted_confidence, 4),
        "top3_predictions": top3_predictions,
    }


# ============================================================
# Run with uvicorn
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

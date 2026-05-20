"""
🚦 German Traffic Sign Classifier - Streamlit App

Interactive web application for classifying German traffic signs
using a trained CNN model.
"""

import io
import os
import pickle

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="German Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered",
)

# Constants
MODELS_DIR = "models"
IMAGE_SIZE = 64
NUM_CLASSES = 43


# ============================================================
# Model Architecture
# ============================================================

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(TrafficSignCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# ============================================================
# Load Model (cached)
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model and class names."""
    # Load class names
    class_names_path = os.path.join(MODELS_DIR, "class_names.pkl")
    if os.path.exists(class_names_path):
        with open(class_names_path, "rb") as f:
            class_names = pickle.load(f)
    else:
        class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

    # Load model
    model_path = os.path.join(MODELS_DIR, "best_model.pth")
    model = TrafficSignCNN(num_classes=NUM_CLASSES)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model, class_names, True
    else:
        return model, class_names, False


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference."""
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_array).unsqueeze(0)
    return tensor


# ============================================================
# App UI
# ============================================================

st.title("🚦 German Traffic Sign Classifier")
st.markdown("Upload a traffic sign image and get instant AI classification using a CNN trained on the GTSRB dataset.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** Custom CNN (3 conv blocks)  
    **Dataset:** GTSRB (50,000+ images)  
    **Classes:** 43 German traffic signs  
    **Accuracy:** 97.15%  
    **Framework:** PyTorch
    """)
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("🔥 PyTorch • 🚀 FastAPI • 🎈 Streamlit")
    st.divider()
    st.markdown("[📂 GitHub Repo](https://github.com/hossain-2002/German-Traffic-Sign-Classification)")

# Load model
model, class_names, model_ok = load_model()

if not model_ok:
    st.error("⚠️ Model file not found. Please ensure `models/best_model.pth` exists.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["png", "jpg", "jpeg"],
    help="Upload a photo of a German traffic sign"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    with col2:
        with st.spinner("Classifying..."):
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze()

            confidence, predicted_class = torch.max(probabilities, dim=0)
            class_id = predicted_class.item()
            conf_value = confidence.item()

            # Main prediction
            st.markdown("### 🎯 Prediction")
            st.success(f"**{class_names[class_id]}**")
            st.metric("Confidence", f"{conf_value * 100:.1f}%")
            st.progress(conf_value)

    # Top 5 predictions
    st.markdown("---")
    st.markdown("### 📊 Top 5 Predictions")
    top5_values, top5_indices = torch.topk(probabilities, k=5)

    for val, idx in zip(top5_values, top5_indices):
        col_name, col_bar = st.columns([2, 3])
        with col_name:
            st.write(f"**{class_names[idx.item()]}**")
        with col_bar:
            st.progress(val.item())
            st.caption(f"{val.item() * 100:.2f}%")

else:
    # Show example when no image uploaded
    st.info("👆 Upload a traffic sign image to get started")
    st.markdown("---")
    st.markdown("### 🏷️ Supported Traffic Sign Classes")
    cols = st.columns(3)
    sample_classes = [
        "🛑 Stop", "⚠️ Yield", "🚫 No entry",
        "🔵 Keep right", "⏱️ Speed limit (30km/h)", "🚧 Road work",
        "↗️ Turn right ahead", "⬆️ Ahead only", "🔄 Roundabout",
    ]
    for i, cls in enumerate(sample_classes):
        cols[i % 3].write(cls)

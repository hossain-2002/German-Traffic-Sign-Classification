# ============================================================
# Dockerfile - German Traffic Sign Classification API
# Production image for serving PyTorch CNN predictions via FastAPI
# ============================================================

# Base image: Python 3.11 slim for smaller footprint
FROM python:3.11-slim

# Install system dependencies required by OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
# Dependencies only re-install when requirements.txt changes
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API application code
COPY api/ ./api/

# Copy trained model and class names
COPY models/ ./models/

# Expose the API port
EXPOSE 8000

# Health check: verify the API is responding every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

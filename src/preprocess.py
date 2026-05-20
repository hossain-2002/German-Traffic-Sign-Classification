"""
GTSRB Dataset Preprocessing Script

Loads raw images, crops to ROI, resizes to 64x64, normalizes,
and saves as numpy arrays ready for PyTorch training.
"""

import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Constants
IMAGE_SIZE = 64
DATA_DIR = "data"
MODELS_DIR = "models"

# GTSRB class names (standard mapping)
CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5t",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5t",
]


def load_and_preprocess_image(path, roi_x1, roi_y1, roi_x2, roi_y2):
    """Load an image, crop to ROI, resize, normalize, and return as CHW array.

    Args:
        path: Path to the image file.
        roi_x1: Left boundary of the ROI.
        roi_y1: Top boundary of the ROI.
        roi_x2: Right boundary of the ROI.
        roi_y2: Bottom boundary of the ROI.

    Returns:
        Numpy array of shape (3, IMAGE_SIZE, IMAGE_SIZE) with values in [0, 1],
        or None if the image cannot be loaded.
    """
    try:
        img = Image.open(path)
        img = img.crop((roi_x1, roi_y1, roi_x2, roi_y2))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img = img.convert("RGB")

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Convert from HWC to CHW format for PyTorch
        img_array = img_array.transpose(2, 0, 1)

        return img_array
    except Exception as e:
        print(f"  [WARNING] Failed to load image: {path} — {e}")
        return None


def process_dataset(csv_path, split_name):
    """Process all images listed in a CSV file.

    Args:
        csv_path: Path to the CSV file (Train.csv or Test.csv).
        split_name: Name of the split ('train' or 'test') for display.

    Returns:
        Tuple of (images, labels) as numpy arrays.
    """
    df = pd.read_csv(csv_path)
    print(f"\nProcessing {split_name} set: {len(df)} images")

    images = []
    labels = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        img_path = os.path.join(DATA_DIR, row["Path"])

        if not os.path.exists(img_path):
            skipped += 1
            continue

        img_array = load_and_preprocess_image(
            img_path,
            roi_x1=int(row["Roi.X1"]),
            roi_y1=int(row["Roi.Y1"]),
            roi_x2=int(row["Roi.X2"]),
            roi_y2=int(row["Roi.Y2"]),
        )

        if img_array is not None:
            images.append(img_array)
            labels.append(int(row["ClassId"]))
        else:
            skipped += 1

    if skipped > 0:
        print(f"  [INFO] Skipped {skipped} images due to errors")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    return X, y


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("  GTSRB Dataset Preprocessing")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Output format: CHW (channels first) for PyTorch")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Process training set
    train_csv = os.path.join(DATA_DIR, "Train.csv")
    X_train, y_train = process_dataset(train_csv, "train")

    # Process test set
    test_csv = os.path.join(DATA_DIR, "Test.csv")
    X_test, y_test = process_dataset(test_csv, "test")

    # Save numpy arrays
    print("\nSaving preprocessed data...")
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    print(f"  Saved: data/X_train.npy, data/y_train.npy")
    print(f"  Saved: data/X_test.npy, data/y_test.npy")

    # Save class names
    class_names_path = os.path.join(MODELS_DIR, "class_names.pkl")
    with open(class_names_path, "wb") as f:
        pickle.dump(CLASS_NAMES, f)
    print(f"  Saved: {class_names_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  X_train shape: {X_train.shape}  dtype: {X_train.dtype}")
    print(f"  y_train shape: {y_train.shape}  dtype: {y_train.dtype}")
    print(f"  X_test shape:  {X_test.shape}  dtype: {X_test.dtype}")
    print(f"  y_test shape:  {y_test.shape}  dtype: {y_test.dtype}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    print(f"  Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()

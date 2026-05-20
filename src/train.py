"""
GTSRB Traffic Sign Classification - PyTorch CNN Training Script

Trains a CNN model on preprocessed GTSRB data (64x64 images, 43 classes).
Includes data augmentation, learning rate scheduling, and evaluation metrics.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Constants
DATA_DIR = "data"
MODELS_DIR = "models"
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 43
IMAGE_SIZE = 64


class TrafficSignDataset(Dataset):
    """PyTorch Dataset for GTSRB traffic sign images.

    Args:
        X: Numpy array of images with shape (N, 3, 64, 64), values in [0, 1].
        y: Numpy array of labels with shape (N,).
        transform: Optional torchvision transforms to apply.
    """

    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


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
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 2: 32 → 64 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 3: 64 → 128 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After 3 MaxPool layers: 64 / 2 / 2 / 2 = 8, but with padding=1 → 8x8
        # Actually: 64→32→16→8, so flatten size = 128 * 8 * 8
        # Wait: let's compute carefully:
        # Input: 64x64 → Conv(pad=1) → 64x64 → MaxPool → 32x32
        # → Conv(pad=1) → 32x32 → MaxPool → 16x16
        # → Conv(pad=1) → 16x16 → MaxPool → 8x8
        # Flatten: 128 * 8 * 8 = 8192
        # But spec says 128*6*6, let's use padding=0 to get 6x6:
        # Input: 64x64 → Conv(3x3, no pad) → 62x62 → MaxPool → 31x31
        # → Conv(3x3, no pad) → 29x29 → MaxPool → 14x14
        # → Conv(3x3, no pad) → 12x12 → MaxPool → 6x6
        # So we need padding=0 for the spec's 128*6*6

        # Redefine with padding=0 to match spec (128*6*6 = 4608)
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

        # Fully connected layers
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


def get_transforms(train=True):
    """Get data augmentation transforms.

    Args:
        train: If True, apply augmentation. If False, no augmentation.

    Returns:
        torchvision.transforms.Compose object.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    return None


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch.

    Args:
        model: The CNN model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use (cuda/cpu).

    Returns:
        Tuple of (average loss, accuracy percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """Evaluate the model on the test set.

    Args:
        model: The CNN model.
        test_loader: DataLoader for test data.
        device: Device to use (cuda/cpu).

    Returns:
        Tuple of (accuracy percentage, all predictions, all true labels).
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save the confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix - GTSRB Classification", fontsize=14)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(range(NUM_CLASSES), fontsize=6, rotation=90)
    ax.set_yticklabels(range(NUM_CLASSES), fontsize=6)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  GTSRB Traffic Sign Classification - Training")
    print("=" * 60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # Create datasets and dataloaders
    train_transform = get_transforms(train=True)
    train_dataset = TrafficSignDataset(X_train, y_train, transform=train_transform)
    test_dataset = TrafficSignDataset(X_test, y_test, transform=None)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Initialize model, loss, optimizer, scheduler
    model = TrafficSignCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Training loop
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_accuracy = 0.0
    print(f"\n{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<12}{'Test Acc':<12}{'LR':<10}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_acc, _, _ = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:<8}{train_loss:<14.4f}{train_acc:<12.2f}{test_acc:<12.2f}{current_lr:<10.6f}"
        )

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))

        scheduler.step()

    # Final evaluation with best model
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (Best Model)")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))
    final_acc, y_pred, y_true = evaluate(model, test_loader, device)
    print(f"\n  Best Test Accuracy: {final_acc:.2f}%")

    # Load class names
    class_names_path = os.path.join(MODELS_DIR, "class_names.pkl")
    if os.path.exists(class_names_path):
        with open(class_names_path, "rb") as f:
            class_names = pickle.load(f)
    else:
        class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

    # Classification report
    print("\n  Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=3, zero_division=0
        )
    )

    # Confusion matrix
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    print("\n  Training complete.")
    print(f"  Best model saved: {os.path.join(MODELS_DIR, 'best_model.pth')}")
    print(f"  Best test accuracy: {best_accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

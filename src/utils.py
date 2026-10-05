"""Utility functions for visualization, metrics, image preprocessing, and parsing."""

from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src import config


def parse_label(label: str) -> Dict[str, str]:
    """Parses a class name string into food item and freshness tier.

    Examples:
        'fresh_banana' -> {'item': 'Banana', 'freshness': 'Fresh'}
        'semifresh_bittermelon' -> {'item': 'Bittermelon', 'freshness': 'Semi-Fresh'}
        'rotten_tomato' -> {'item': 'Tomato', 'freshness': 'Rotten'}
    """
    clean = label.lower().strip()

    # Identify freshness
    if "semifresh" in clean or "semi_fresh" in clean:
        freshness = "Semi-Fresh"
        clean = clean.replace("semifresh", "").replace("semi_fresh", "")
    elif "fresh" in clean:
        freshness = "Fresh"
        clean = clean.replace("fresh", "")
    elif "rotten" in clean:
        freshness = "Rotten"
        clean = clean.replace("rotten", "")
    else:
        freshness = "Unknown"

    clean = clean.replace("_", "").strip()

    # Identify item
    matched_item = "Unknown"
    for item in config.FOOD_ITEMS:
        if item.lower() in clean or clean in item.lower():
            matched_item = item
            break

    return {
        "raw_class": label,
        "item": matched_item,
        "freshness": freshness,
    }


def preprocess_image(
    image_input,
    target_size: Tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """Preprocesses an input image (filepath, numpy array, or PIL Image) for model inference.

    Returns:
        np.ndarray with shape (1, height, width, 3) ready for model prediction.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Failed to load image at: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img = image_input
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    batch_img = np.expand_dims(resized, axis=0)
    return batch_img


def plot_training_history(history, save_path: Optional[Path] = None) -> None:
    """Plots training and validation accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", lw=2)
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="#10b981", lw=2, linestyle="--")
    axes[0].set_title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Loss Plot
    axes[1].plot(epochs_range, loss, label="Training Loss", color="#dc2626", lw=2)
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#f59e0b", lw=2, linestyle="--")
    axes[1].set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """Plots and saves a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized Confusion Matrix (24 Classes)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

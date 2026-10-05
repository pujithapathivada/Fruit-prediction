"""Dataset preview script for Food Freshness Detection.

Loads the training dataset using TensorFlow/Keras, displays batch shapes,
and renders a 3x3 grid of sample images with their class labels saved to
notebooks/dataset_preview.png.

Safety: Read-only inspection. Does not modify dataset and does not train any model.
"""

import os
import sys
from pathlib import Path

# Suppress excessive TensorFlow INFO/WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_PREVIEW_PATH = NOTEBOOKS_DIR / "dataset_preview.png"


def preview_training_dataset():
    """Loads dataset/train, prints dataset specs, and saves a 3x3 sample grid."""
    print("=" * 70)
    print(" Food Freshness Dataset - Training Batch Preview")
    print("=" * 70)
    print(f"Loading training data from: {TRAIN_DIR}\n")

    if not TRAIN_DIR.exists():
        print(f"[ERROR] Training directory not found: {TRAIN_DIR}")
        sys.exit(1)

    # 1. Load dataset/train with specified parameters
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(TRAIN_DIR),
        image_size=(224, 224),
        batch_size=32,
        shuffle=True,
        seed=42,
    )

    # 2. Extract class names and counts
    class_names = train_ds.class_names
    num_classes = len(class_names)
    num_batches = len(train_ds)

    # 3. Print detected class names
    print("\nDetected Class Names:")
    print("-" * 50)
    for idx, cname in enumerate(class_names, start=1):
        print(f"  {idx:>2}. {cname}")
    print("-" * 50)

    # 4. Take the first batch to inspect shapes
    for images, labels in train_ds.take(1):
        image_batch_shape = images.shape
        label_batch_shape = labels.shape

        print(f"\nDataset Batch Specifications:")
        print(f"• Number of classes : {num_classes}")
        print(f"• Number of batches : {num_batches}")
        print(f"• Image batch shape : {image_batch_shape}")
        print(f"• Label batch shape : {label_batch_shape}")

        # 5. Display 9 sample training images in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(11, 11))
        fig.suptitle(
            "AgriFreshNET Training Sample Preview (3x3 Grid)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        for i in range(9):
            row = i // 3
            col = i % 3
            ax = axes[row, col]

            # Convert tensor image to uint8 format for matplotlib display
            img_array = images[i].numpy().astype("uint8")
            label_idx = int(labels[i].numpy())
            class_label = class_names[label_idx]

            ax.imshow(img_array)
            # 6. Display class name above each image
            ax.set_title(class_label, fontsize=10, fontweight="bold", pad=6)
            ax.axis("off")

        plt.tight_layout()

        # 7. Save preview image to notebooks/dataset_preview.png
        NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_PREVIEW_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"\n[OK] 3x3 Preview grid successfully saved to:")
        print(f"     {OUTPUT_PREVIEW_PATH}")
        break

    print("=" * 70)


def main():
    preview_training_dataset()


if __name__ == "__main__":
    main()

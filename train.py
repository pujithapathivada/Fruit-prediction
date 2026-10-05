"""Food Freshness Detection - Model Training Pipeline.

Trains a MobileNetV2 Transfer Learning model to classify 24 categories
from the AgriFreshNET dataset into Fresh, Semi-Fresh, and Rotten states.

Features:
- Dynamically loads 24 classes from dataset/train
- Data augmentation: RandomFlip, RandomRotation, RandomZoom
- MobileNetV2 backbone (ImageNet pre-trained, initially frozen)
- GlobalAveragePooling2D + Dropout(0.3) + Softmax Dense layer
- EarlyStopping & ModelCheckpoint callbacks
- Evaluates on dataset/test
- Exports training history JSON and plots to notebooks/training_history.png
"""

import json
import os
import sys
from pathlib import Path

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Safe UTF-8 console output for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ==========================================
# 1. Project Paths and Hyperparameters
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MODEL_SAVE_PATH = MODELS_DIR / "freshness_model.keras"
HISTORY_JSON_PATH = MODELS_DIR / "training_history.json"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
PLOT_SAVE_PATH = NOTEBOOKS_DIR / "training_history.png"

# Ensure target output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
LEARNING_RATE = 0.0001
MAX_EPOCHS = 15
SEED = 42


def build_data_augmentation() -> tf.keras.Sequential:
    """Builds the data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.1, seed=SEED),
            tf.keras.layers.RandomZoom(0.1, seed=SEED),
        ],
        name="data_augmentation",
    )


def build_mobilenet_model(num_classes: int) -> tf.keras.Model:
    """Constructs the MobileNetV2 transfer learning model."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")

    # Step 1: Data Augmentation
    x = build_data_augmentation()(inputs)

    # Step 2: MobileNetV2 Preprocessing (scales pixel values from [0, 255] to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Step 3: MobileNetV2 Backbone (pre-trained on ImageNet, initially frozen)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # Run base model in inference mode
    x = base_model(x, training=False)

    # Step 4: Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="freshness_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="FoodFreshness_MobileNetV2",
    )
    return model


def plot_and_save_history(history_dict: dict, save_path: Path) -> None:
    """Generates and saves accuracy and loss curves."""
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs, history_dict["accuracy"], label="Training Accuracy", color="#2563eb", lw=2)
    ax1.plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy", color="#16a34a", lw=2, linestyle="--")
    ax1.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Loss Plot
    ax2.plot(epochs, history_dict["loss"], label="Training Loss", color="#dc2626", lw=2)
    ax2.plot(epochs, history_dict["val_loss"], label="Validation Loss", color="#ea580c", lw=2, linestyle="--")
    ax2.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Training curves saved to: {save_path}")


def train() -> None:
    """Main training workflow."""
    print("=" * 75)
    print(" Food Freshness Detection - Deep Learning Training Pipeline")
    print("=" * 75)

    # Verify dataset directories exist
    for dir_path, label in [(TRAIN_DIR, "Train"), (VAL_DIR, "Validation"), (TEST_DIR, "Test")]:
        if not dir_path.exists():
            raise FileNotFoundError(f"{label} dataset directory not found at: {dir_path}")

    # ==========================================
    # 2. Load Datasets using image_dataset_from_directory
    # ==========================================
    print("\n[1/5] Loading datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(TRAIN_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(VAL_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(TEST_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Dynamically extract class names from the dataset
    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"\n[+] Successfully loaded {num_classes} classes automatically:")
    for idx, cname in enumerate(class_names, start=1):
        print(f"    {idx:>2}. {cname}")

    # Save class indices for future inference & UI
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print(f"[+] Class indices saved to: {CLASS_INDICES_PATH}")

    # Optimize datasets pipeline with prefetching
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    # ==========================================
    # 3. Build & Compile MobileNetV2 Model
    # ==========================================
    print(f"\n[2/5] Constructing MobileNetV2 transfer learning model for {num_classes} classes...")
    model = build_mobilenet_model(num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    model.summary()

    # ==========================================
    # 4. Training with Callbacks
    # ==========================================
    print(f"\n[3/5] Starting training for maximum {MAX_EPOCHS} epochs...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
    )

    # Ensure best weights are preserved and saved
    model.save(str(MODEL_SAVE_PATH))
    print(f"\n[+] Final best model saved to: {MODEL_SAVE_PATH}")

    # ==========================================
    # 5. Evaluate on Test Dataset
    # ==========================================
    print("\n[4/5] Evaluating model performance on unseen test dataset...")
    test_results = model.evaluate(test_ds, verbose=1)
    test_loss, test_accuracy = test_results[0], test_results[1]

    # Save training history to JSON (convert float32 to standard python float)
    history_dict = {
        key: [float(val) for val in values]
        for key, values in history.history.items()
    }
    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)
    print(f"[+] Training history saved to: {HISTORY_JSON_PATH}")

    # ==========================================
    # 6. Save Plots & Final Summary Message
    # ==========================================
    print("\n[5/5] Generating accuracy and loss visualizations...")
    plot_and_save_history(history_dict, PLOT_SAVE_PATH)

    best_val_accuracy = max(history_dict["val_accuracy"])

    print("\n" + "=" * 75)
    print(" TRAINING SUMMARY")
    print("=" * 75)
    print("Training completed")
    print(f"Total Epochs Run       : {len(history_dict['loss'])} / {MAX_EPOCHS}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy * 100:.2f}%)")
    print(f"Test Accuracy           : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test Loss               : {test_loss:.4f}")
    print(f"Model saved path        : {MODEL_SAVE_PATH}")
    print("\nClass Names Detected and Trained:")
    for idx, cname in enumerate(class_names, start=1):
        print(f"  {idx:>2}. {cname}")
    print("=" * 75)


if __name__ == "__main__":
    train()

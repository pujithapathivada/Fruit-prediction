"""Configuration parameters for Food Freshness Detection."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "food_freshness_best_model.keras"

# Create directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Class Definitions
FOOD_ITEMS = [
    "Banana",
    "Orange",
    "Pineapple",
    "Tomato",
    "Eggplant",
    "Cucumber",
    "Papaya",
    "Bittermelon",
]

FRESHNESS_LEVELS = [
    "Fresh",
    "Semi-Fresh",
    "Rotten",
]

# Canonical 24 Classes in AgriFreshNET Processed_Data
CLASS_NAMES = [
    # Banana
    "fresh_banana",
    "semifresh_banana",
    "rotten_banana",
    # Orange
    "fresh_orange",
    "semifresh_orange",
    "rotten_orange",
    # Pineapple
    "fresh_pineapple",
    "semifresh_pineapple",
    "rotten_pineapple",
    # Tomato
    "fresh_tomato",
    "semifresh_tomato",
    "rotten_tomato",
    # Eggplant
    "fresh_eggplant",
    "semifresh_eggplant",
    "rotten_eggplant",
    # Cucumber
    "fresh_cucumber",
    "semifresh_cucumber",
    "rotten_cucumber",
    # Papaya
    "fresh_papaya",
    "semifresh_papaya",
    "rotten_papaya",
    # Bittermelon
    "fresh_bittermelon",
    "semifresh_bittermelon",
    "rotten_bittermelon",
]

NUM_CLASSES = len(CLASS_NAMES)  # 24

# Image Specifications & Training Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

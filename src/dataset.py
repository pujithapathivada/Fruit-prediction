"""Dataset loading, splitting, and augmentation utilities."""

from pathlib import Path
from typing import Tuple, Optional, List
import tensorflow as tf
from src import config


def get_data_augmentation() -> tf.keras.Sequential:
    """Returns a Keras Sequential layer sequence for data augmentation."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def load_datasets(
    data_dir: Optional[Path] = None,
    img_size: Tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    validation_split: float = config.VALIDATION_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Loads and prepares training and validation tf.data.Dataset instances.

    Args:
        data_dir: Path to directory containing class subdirectories.
        img_size: Target image dimensions (height, width).
        batch_size: Batch size for training and evaluation.
        validation_split: Fraction of dataset reserved for validation.
        seed: Random seed for reproducible splitting.

    Returns:
        (train_ds, val_ds, class_names)
    """
    directory = Path(data_dir) if data_dir else config.DATASET_DIR

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    # Performance optimization: cache and prefetch
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names

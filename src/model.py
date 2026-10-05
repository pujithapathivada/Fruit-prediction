"""Deep Learning model architecture definitions using TensorFlow/Keras."""

from typing import Tuple
import tensorflow as tf
from src import config
from src.dataset import get_data_augmentation


def build_transfer_model(
    num_classes: int = config.NUM_CLASSES,
    input_shape: Tuple[int, int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    backbone_name: str = "MobileNetV2",
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Builds a transfer-learning model for food freshness classification.

    Args:
        num_classes: Total number of target classes (24).
        input_shape: Input image dimensions.
        backbone_name: Backbone architecture ("MobileNetV2" or "EfficientNetB0").
        freeze_backbone: Whether to freeze base backbone weights initially.
        dropout_rate: Dropout rate before the classification head.

    Returns:
        Compiled or uncompiled tf.keras.Model.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Data augmentation block
    x = get_data_augmentation()(inputs)

    # Backbone selection and preprocessing
    if backbone_name.lower() == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name.lower() == "efficientnetb0":
        # EfficientNet has built-in rescaling
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base_model.trainable = not freeze_backbone
    x = base_model(x, training=False)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="freshness_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"FoodFreshness_{backbone_name}")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.Model:
    """Compiles the Keras model with Adam optimizer and classification metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return model

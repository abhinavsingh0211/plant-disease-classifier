"""
Model definitions.

Two families:

1. `build_cnn`            – a compact CNN trained from scratch (the baseline that
                           justifies why we reach for transfer learning).
2. `build_transfer_model`– MobileNetV2 / ResNet50 / EfficientNetB0 pretrained on
                           ImageNet with a fresh classification head.

Key design choice: the backbone's required pixel pre-processing (`preprocess_input`)
is baked **into the model graph**. The saved `.keras` file therefore accepts raw
uint8 [0, 255] images, so the Streamlit app never has to re-implement (and risk
mismatching) the training-time normalisation — a very common source of
"great in the notebook, wrong in production" bugs.
"""
from __future__ import annotations

import tensorflow as tf

from .config import CFG
from .data import build_augmenter

layers = tf.keras.layers

# backbone name -> (constructor, preprocessing fn)
_BACKBONES = {
    "mobilenetv2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "resnet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
    ),
    "efficientnetb0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
}


def get_backbone(model: tf.keras.Model):
    """Return the nested pretrained backbone (a Keras Model used as a layer), or
    None for the from-scratch CNN.

    Identified by *content* rather than name: the augmentation pipeline is also a
    nested `Sequential` (a Model subclass), so we specifically pick the nested
    model that contains convolutional layers — i.e. the real backbone. This is
    more robust than a name lookup, which Keras 3 doesn't reliably honour for
    pretrained models built via `keras.applications`.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(
            isinstance(sub, tf.keras.layers.Conv2D) for sub in layer.layers
        ):
            return layer
    return None


def build_cnn(num_classes: int = None, input_shape=(224, 224, 3)) -> tf.keras.Model:
    """A small VGG-style CNN trained from scratch — the baseline."""
    num_classes = num_classes or CFG.num_classes
    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmenter()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(CFG.dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="scratch_cnn")
    return model


def build_transfer_model(
    backbone: str = None,
    num_classes: int = None,
    input_shape=(224, 224, 3),
) -> tf.keras.Model:
    """Pretrained backbone (frozen) + augmentation + preprocessing + new head."""
    backbone = (backbone or CFG.backbone).lower()
    num_classes = num_classes or CFG.num_classes
    if backbone == "cnn":
        return build_cnn(num_classes, input_shape)
    if backbone not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Options: {list(_BACKBONES)}")

    constructor, preprocess = _BACKBONES[backbone]
    base = constructor(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False           # phase 1: feature extraction (backbone frozen)

    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmenter()(inputs)
    x = preprocess(x)
    # training=False keeps the backbone's BatchNorm layers in inference mode,
    # which matters both when frozen and later when fine-tuning.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(CFG.dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone}_transfer")
    return model


def unfreeze_for_finetuning(model: tf.keras.Model, fine_tune_at: float = None) -> int:
    """Unfreeze the top fraction of the backbone for phase-2 fine-tuning.

    BatchNorm layers are deliberately left frozen (in inference mode) — updating
    their running statistics on a small fine-tuning run tends to hurt.
    Returns the number of newly-trainable layers.
    """
    fine_tune_at = CFG.fine_tune_at if fine_tune_at is None else fine_tune_at
    base = get_backbone(model)
    if base is None:
        # from-scratch CNN has no separate backbone; nothing to do.
        return 0

    base.trainable = True
    cutoff = int(len(base.layers) * fine_tune_at)
    trainable = 0
    for i, layer in enumerate(base.layers):
        if i < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable += 1
    return trainable


def compile_model(model: tf.keras.Model, lr: float):
    """Compile with Adam + sparse categorical cross-entropy.

    We keep integer labels end-to-end (label_mode="int"), so the sparse loss is
    the correct pairing and lets the whole train/eval pipeline stay simple.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

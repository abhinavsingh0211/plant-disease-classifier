"""
Hyperparameter tuning with KerasTuner (Hyperband).

Searches the classification-head hyperparameters that matter most for transfer
learning — dropout, an optional dense bottleneck, and the head learning rate —
while the ImageNet backbone stays frozen (so each trial is cheap). Hyperband
allocates most of the budget to the promising configs via successive halving.

Run
---
    python -m src.tune --max-epochs 10
The best hyperparameters are printed and saved to models/best_hparams.json.
"""
from __future__ import annotations

import argparse

import tensorflow as tf

from .config import CFG
from .data import build_augmenter, make_datasets
from .utils import save_json, set_seed

layers = tf.keras.layers


def _build_tunable(hp, num_classes: int):
    """Model-building function consumed by KerasTuner."""
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(*CFG.image_size, 3)
    )
    base.trainable = False

    dropout = hp.Float("dropout", 0.2, 0.5, step=0.1)
    dense_units = hp.Choice("dense_units", [0, 128, 256])   # 0 = no bottleneck
    lr = hp.Choice("lr", [1e-3, 5e-4, 1e-4])

    inputs = tf.keras.Input(shape=(*CFG.image_size, 3))
    x = build_augmenter()(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    if dense_units:
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def tune(max_epochs: int = 10, factor: int = 3, data_root=None):
    import keras_tuner as kt

    set_seed(CFG.seed)
    CFG.ensure_dirs()

    train_ds, val_ds, _, class_names = make_datasets(data_root)

    tuner = kt.Hyperband(
        lambda hp: _build_tunable(hp, len(class_names)),
        objective="val_accuracy",
        max_epochs=max_epochs,
        factor=factor,
        directory=str(CFG.models_dir / "kt"),
        project_name="plant_disease",
        overwrite=True,
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )
    tuner.search(train_ds, validation_data=val_ds, callbacks=[stop_early])

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best = {
        "dropout": best_hp.get("dropout"),
        "dense_units": best_hp.get("dense_units"),
        "lr": best_hp.get("lr"),
    }
    save_json(best, CFG.models_dir / "best_hparams.json")
    print("\n[tune] best hyperparameters:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"[tune] saved -> {CFG.models_dir/'best_hparams.json'}")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tune head hyperparameters (Hyperband).")
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--data-root", default=None)
    args = p.parse_args()
    tune(max_epochs=args.max_epochs, factor=args.factor, data_root=args.data_root)

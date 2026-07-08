"""
Training entrypoint (two-phase transfer learning).

Phase 1 – feature extraction: freeze the ImageNet backbone, train only the new
          classification head with a normal learning rate.
Phase 2 – fine-tuning: unfreeze the top of the backbone and continue with a tiny
          learning rate so the pretrained filters adapt to leaf textures without
          being destroyed.

Run
---
    python -m src.train                      # MobileNetV2 (default)
    python -m src.train --backbone resnet50
    python -m src.train --backbone cnn --no-fine-tune   # from-scratch baseline
"""
from __future__ import annotations

import argparse
import time

import tensorflow as tf

from .config import CFG
from .data import make_datasets
from .models import (
    build_transfer_model,
    compile_model,
    unfreeze_for_finetuning,
)
from .utils import plot_history, save_json, set_seed


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=CFG.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


def train(
    backbone: str = None,
    data_root=None,
    fine_tune: bool = None,
    head_epochs: int = None,
    fine_tune_epochs: int = None,
):
    backbone = backbone or CFG.backbone
    fine_tune = CFG.fine_tune if fine_tune is None else fine_tune
    head_epochs = head_epochs or CFG.head_epochs
    fine_tune_epochs = fine_tune_epochs or CFG.fine_tune_epochs

    set_seed(CFG.seed)
    CFG.ensure_dirs()

    train_ds, val_ds, test_ds, class_names = make_datasets(data_root)
    save_json(class_names, CFG.class_names_path)

    model = build_transfer_model(backbone=backbone, num_classes=len(class_names))
    compile_model(model, lr=CFG.head_lr)
    model.summary()

    histories = []

    # ---------------- Phase 1: feature extraction ---------------- #
    print("\n=== Phase 1: training classification head ===")
    t0 = time.time()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        callbacks=_callbacks(),
    )
    histories.append(h1.history)

    # ---------------- Phase 2: fine-tuning ----------------------- #
    if fine_tune and backbone != "cnn":
        n = unfreeze_for_finetuning(model)
        print(f"\n=== Phase 2: fine-tuning {n} backbone layers ===")
        compile_model(model, lr=CFG.fine_tune_lr)   # recompile after unfreezing
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=_callbacks(),
        )
        histories.append(h2.history)

    train_minutes = (time.time() - t0) / 60
    print(f"\n[train] total training time: {train_minutes:.1f} min")

    # ---------------- Persist artefacts -------------------------- #
    model.save(CFG.model_path)
    print(f"[train] saved model -> {CFG.model_path}")

    if len(histories) >= 1:
        plot_history(histories, CFG.assets_dir / "accuracy_loss_graph.png")
        print(f"[train] saved training curves -> {CFG.assets_dir/'accuracy_loss_graph.png'}")

    # A quick held-out test number so `train` alone gives a headline metric.
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[train] held-out test accuracy: {test_acc:.4f}")

    return model, (train_ds, val_ds, test_ds), class_names


def _parse_args():
    p = argparse.ArgumentParser(description="Train the plant disease classifier.")
    p.add_argument("--backbone", default=CFG.backbone,
                   choices=["mobilenetv2", "resnet50", "efficientnetb0", "cnn"])
    p.add_argument("--data-root", default=None,
                   help="Path to an already-downloaded dataset (skips kagglehub).")
    p.add_argument("--head-epochs", type=int, default=CFG.head_epochs)
    p.add_argument("--fine-tune-epochs", type=int, default=CFG.fine_tune_epochs)
    p.add_argument("--no-fine-tune", action="store_true", help="Skip phase 2.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        backbone=args.backbone,
        data_root=args.data_root,
        fine_tune=not args.no_fine_tune,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
    )

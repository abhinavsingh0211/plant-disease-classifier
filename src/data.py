"""
Data loading for the New Plant Diseases Dataset.

Responsibilities
----------------
* download the dataset with kagglehub (cached after the first run),
* transparently handle the dataset's awkward *nested* folder layout
  (`.../New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train`),
* build fast `tf.data` pipelines (cache + prefetch + AUTOTUNE),
* carve a held-out **test** split out of the official `valid` folder so we can
  report metrics on data the model never saw during training or model selection,
* compute class weights (the set is roughly balanced, but this is defensive and
  a good thing to show awareness of).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import CFG

AUTOTUNE = tf.data.AUTOTUNE


def download_dataset() -> Path:
    """Download via kagglehub and return the cache path. No API token needed
    for public datasets when running on Colab/Kaggle."""
    import kagglehub

    path = Path(kagglehub.dataset_download(CFG.kaggle_dataset))
    print(f"[data] dataset cached at: {path}")
    return path


def _find_split_dir(root: Path, split: str) -> Path:
    """Locate the `train` or `valid` directory no matter how deeply the archive
    nested it. We look for a directory named `split` that contains sub-folders
    (the class directories)."""
    candidates = [p for p in root.rglob(split) if p.is_dir()]
    # Prefer the one that actually holds class sub-directories.
    for c in sorted(candidates, key=lambda p: len(p.parts)):
        if any(child.is_dir() for child in c.iterdir()):
            return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find a '{split}' directory under {root}. "
        f"Contents: {[p.name for p in root.iterdir()]}"
    )


def locate_data(root: Path | None = None) -> tuple[Path, Path]:
    """Return `(train_dir, valid_dir)` for the dataset."""
    root = Path(root) if root is not None else download_dataset()
    train_dir = _find_split_dir(root, "train")
    valid_dir = _find_split_dir(root, "valid")
    print(f"[data] train: {train_dir}")
    print(f"[data] valid: {valid_dir}")
    return train_dir, valid_dir


def _dataset_from_dir(directory: Path, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=CFG.image_size,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        seed=CFG.seed,
    )


def make_datasets(root: Path | None = None, test_fraction: float = 0.5):
    """Build train / val / test datasets.

    The official dataset only ships `train` + `valid` (+ 33 loose demo images).
    We split the official `valid` set in half -> a validation set (for early
    stopping / model selection) and a **held-out test set** (for the final,
    unbiased metrics reported in the README).

    Returns
    -------
    train_ds, val_ds, test_ds, class_names
    """
    train_dir, valid_dir = locate_data(root)

    train_ds = _dataset_from_dir(train_dir, shuffle=True)
    class_names = train_ds.class_names

    valid_ds_full = _dataset_from_dir(valid_dir, shuffle=True)
    # Sanity check: the val folder must expose the same classes in the same order.
    assert valid_ds_full.class_names == class_names, "train/valid class mismatch"

    n_val_batches = int(valid_ds_full.cardinality().numpy())
    n_test_batches = int(n_val_batches * test_fraction)
    test_ds = valid_ds_full.take(n_test_batches)
    val_ds = valid_ds_full.skip(n_test_batches)

    print(f"[data] classes: {len(class_names)} | "
          f"val batches: {n_val_batches - n_test_batches} | "
          f"test batches: {n_test_batches}")

    # Performance: prefetch the next batch while the GPU works on the current one.
    # We deliberately do NOT .cache() — the decoded training set (~70k images at
    # 224x224x3) is far too large to hold in RAM and would OOM Colab. Prefetching
    # alone overlaps JPEG decode with GPU compute, which is what matters here.
    # (image_dataset_from_directory already reshuffles the file order each epoch.)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_augmenter() -> tf.keras.Sequential:
    """Light, label-preserving augmentation. Kept as Keras layers so it becomes
    part of the model graph and is automatically inert at inference time."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def compute_class_weights(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Inverse-frequency class weights, keyed by class index."""
    counts = np.array(
        [len(list((train_dir / name).glob("*"))) for name in class_names],
        dtype=np.float64,
    )
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}

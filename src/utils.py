"""Small shared helpers: reproducibility, label prettifying, and plotting."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Make a run as reproducible as a GPU allows."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # keras 3 utility, present in recent TF builds
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


def prettify_label(raw: str) -> str:
    """`Tomato___Late_blight` -> `Tomato — Late blight` for display."""
    if "___" in raw:
        crop, disease = raw.split("___", 1)
    else:
        crop, disease = raw, ""
    crop = crop.replace("_", " ").replace("(including sour)", "").strip()
    disease = disease.replace("_", " ").strip()
    if disease.lower() == "healthy":
        return f"{crop} — Healthy"
    return f"{crop} — {disease}" if disease else crop


def is_healthy(raw: str) -> bool:
    return raw.strip().lower().endswith("healthy")


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def plot_history(history_dicts, out_path: str | Path):
    """Plot stitched accuracy/loss curves across the head + fine-tune phases.

    `history_dicts` is a list of keras History.history dicts (one per phase).
    """
    import matplotlib.pyplot as plt

    acc, val_acc, loss, val_loss = [], [], [], []
    for h in history_dicts:
        acc += h.get("accuracy", [])
        val_acc += h.get("val_accuracy", [])
        loss += h.get("loss", [])
        val_loss += h.get("val_loss", [])

    epochs = range(1, len(acc) + 1)
    boundary = len(history_dicts[0].get("accuracy", [])) if history_dicts else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, acc, "o-", label="train")
    ax1.plot(epochs, val_acc, "o-", label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, loss, "o-", label="train")
    ax2.plot(epochs, val_loss, "o-", label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.set_ylabel("loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    if boundary and boundary < len(acc):
        for ax in (ax1, ax2):
            ax.axvline(boundary + 0.5, color="grey", ls="--", alpha=0.7)
            ax.text(boundary + 0.6, ax.get_ylim()[0], " fine-tuning →",
                    fontsize=9, color="grey", va="bottom")

    fig.suptitle("Training history", fontsize=14, fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path

"""
Single-image inference — the one place prediction logic lives, imported by both
the Streamlit app and the CLI so they can never drift apart.

Because pre-processing is baked into the saved model, inference is refreshingly
simple: resize to 224x224, keep raw [0, 255] pixels, and call the model.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CFG, CLASS_NAMES
from .utils import is_healthy, load_json, prettify_label


@lru_cache(maxsize=1)
def load_model_and_labels(model_path: str = None):
    """Load (and cache) the Keras model plus the class-name list."""
    import tensorflow as tf

    model_path = model_path or str(CFG.model_path)
    model = tf.keras.models.load_model(model_path)
    try:
        class_names = load_json(CFG.class_names_path)
    except FileNotFoundError:
        class_names = CLASS_NAMES  # fallback to the canonical ordering
    return model, class_names


def load_image(path_or_pil, target_size=None) -> np.ndarray:
    """Return a (H, W, 3) uint8 RGB array from a path or PIL image."""
    from PIL import Image

    target_size = target_size or CFG.image_size
    img = Image.open(path_or_pil) if not hasattr(path_or_pil, "size") else path_or_pil
    img = img.convert("RGB").resize(target_size)
    return np.asarray(img, dtype=np.uint8)


def predict(path_or_pil, model=None, class_names=None, top_k: int = 3) -> dict:
    """Classify one leaf image.

    Returns a dict with the top prediction, whether it reads as healthy, and the
    top-k (label, confidence) list — plus the raw batched array so callers (the
    app) can reuse it for Grad-CAM without re-loading the image.
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_labels()

    rgb = load_image(path_or_pil)
    batch = rgb.astype("float32")[None, ...]      # (1, H, W, 3), raw pixels
    probs = model.predict(batch, verbose=0)[0]

    order = np.argsort(probs)[::-1][:top_k]
    top = [
        {
            "label": class_names[i],
            "pretty": prettify_label(class_names[i]),
            "confidence": float(probs[i]),
        }
        for i in order
    ]
    best = class_names[int(order[0])]
    return {
        "prediction": prettify_label(best),
        "raw_label": best,
        "healthy": is_healthy(best),
        "confidence": float(probs[int(order[0])]),
        "top_k": top,
        "image": rgb,
        "batch": batch,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict the disease for one leaf image.")
    p.add_argument("image", type=str, help="Path to a leaf image.")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    result = predict(Path(args.image), top_k=args.top_k)
    status = "🌿 healthy" if result["healthy"] else "⚠️  diseased"
    print(f"\nPrediction : {result['prediction']}  ({status})")
    print(f"Confidence : {result['confidence']*100:.1f}%\n")
    print("Top predictions:")
    for r in result["top_k"]:
        print(f"  {r['confidence']*100:6.2f}%  {r['pretty']}")

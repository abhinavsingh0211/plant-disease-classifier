"""
Model evaluation & reporting.

Produces the artefacts a reviewer expects to see:
* overall accuracy / macro-precision / macro-recall / macro-F1 on the held-out
  test split,
* a full per-class classification report,
* a confusion matrix figure,
* the 15 hardest (most-confused) class pairs,
all saved to `models/metrics.json` and `assets/`.

Run
---
    python -m src.evaluate                    # loads models/plant_disease_model.keras
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import CFG
from .data import make_datasets
from .utils import load_json, prettify_label, save_json


def collect_predictions(model, dataset):
    """Return (y_true, y_pred, y_prob) numpy arrays over a tf.data dataset."""
    y_true, y_prob = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob


def top_k_accuracy(y_true, y_prob, k: int = 3) -> float:
    topk = np.argsort(y_prob, axis=1)[:, -k:]
    hits = [y_true[i] in topk[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def plot_confusion_matrix(cm, class_names, out_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Row-normalise so colour = recall per true class (robust to any imbalance).
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    short = [c.split("___")[0][:10] + "…" if len(c) > 12 else c.split("___")[0]
             for c in class_names]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, cmap="viridis", square=True, cbar_kws={"shrink": 0.6},
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7); plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def most_confused_pairs(cm, class_names, n: int = 15):
    """Return the n off-diagonal (true -> predicted) confusions with highest count."""
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    return [
        {"count": c, "true": prettify_label(t), "predicted": prettify_label(p)}
        for c, t, p in pairs[:n]
    ]


def evaluate(model=None, test_ds=None, class_names=None, data_root=None):
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    CFG.ensure_dirs()

    if model is None:
        model = tf.keras.models.load_model(CFG.model_path)
    if class_names is None:
        class_names = load_json(CFG.class_names_path)
    if test_ds is None:
        _, _, test_ds, class_names = make_datasets(data_root)

    y_true, y_pred, y_prob = collect_predictions(model, test_ds)

    metrics = {
        "backbone": CFG.backbone,
        "n_test_images": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "top3_accuracy": top_k_accuracy(y_true, y_prob, k=3),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["per_class"] = {
        class_names[i]: {
            "precision": round(report[class_names[i]]["precision"], 4),
            "recall": round(report[class_names[i]]["recall"], 4),
            "f1": round(report[class_names[i]]["f1-score"], 4),
            "support": int(report[class_names[i]]["support"]),
        }
        for i in range(len(class_names))
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics["most_confused"] = most_confused_pairs(cm, class_names)

    save_json(metrics, CFG.metrics_path)
    plot_confusion_matrix(cm, class_names, CFG.assets_dir / "confusion_matrix.png")

    # ---- console summary ---- #
    print("\n================ Test-set metrics ================")
    print(f"images        : {metrics['n_test_images']}")
    print(f"accuracy      : {metrics['accuracy']:.4f}")
    print(f"top-3 accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"macro F1      : {metrics['macro_f1']:.4f}")
    print(f"weighted F1   : {metrics['weighted_f1']:.4f}")
    print(f"\nsaved -> {CFG.metrics_path}")
    print(f"saved -> {CFG.assets_dir/'confusion_matrix.png'}")
    print("\nHardest confusions:")
    for p in metrics["most_confused"][:5]:
        print(f"  {p['count']:>3}x  {p['true']}  ->  {p['predicted']}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the trained model.")
    p.add_argument("--data-root", default=None)
    args = p.parse_args()
    evaluate(data_root=args.data_root)

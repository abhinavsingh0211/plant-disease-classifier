"""
Grad-CAM (Gradient-weighted Class Activation Mapping).

Given a leaf image and the trained model, this highlights the regions the network
relied on — it should light up the actual lesions / diseased patches. That turns
the classifier from a black box into something a domain user (or an interviewer)
can sanity-check.

The hard part is that our transfer model wraps the pretrained network as a single
*nested* sub-model. You cannot build a Grad-CAM ``Model(model.input,
[backbone.output, model.output])`` in Keras 3 — the backbone's output tensor lives
in its own internal graph and isn't routable from the outer input ("graph
disconnected"). So instead of graph surgery we run a **manual eager forward pass**
that splits the network at the backbone: raw image -> preprocess -> backbone
(watch this feature map) -> classification head -> class score, then take the
gradient of the score w.r.t. the watched feature map. This is robust and works
for any nested backbone. For the from-scratch CNN (no nested sub-model) we use
the standard functional Grad-CAM model.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .models import get_backbone


def _preprocess_for(backbone: tf.keras.Model):
    """Return the ImageNet preprocessing that matches a given backbone, inferred
    from its layer name. Mirrors what ``build_transfer_model`` bakes into the
    graph so the manual forward pass is numerically identical to model.predict."""
    name = (backbone.name or "").lower()
    if "mobilenet" in name:
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if "resnet" in name:
        return tf.keras.applications.resnet50.preprocess_input
    if "efficientnet" in name:
        return tf.keras.applications.efficientnet.preprocess_input
    return lambda x: x


def _last_conv_layer(model: tf.keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def make_gradcam_heatmap(
    img_array: np.ndarray, model: tf.keras.Model, pred_index: int | None = None
) -> tuple[np.ndarray, int]:
    """Compute a [0, 1] Grad-CAM heatmap for a single image.

    Parameters
    ----------
    img_array : float array of shape (1, H, W, 3) with **raw** pixel values in
        [0, 255] — preprocessing is handled internally.
    pred_index : class index to explain; defaults to the top prediction.

    Returns
    -------
    (heatmap, pred_index)
    """
    img = tf.convert_to_tensor(img_array, dtype=tf.float32)
    backbone = get_backbone(model)

    if backbone is not None:
        # ---- transfer model: manual forward, split at the nested backbone ---- #
        preprocess = _preprocess_for(backbone)
        head_layers = model.layers[model.layers.index(backbone) + 1:]
        with tf.GradientTape() as tape:
            x = preprocess(img)
            features = backbone(x, training=False)
            tape.watch(features)
            h = features
            for layer in head_layers:
                h = layer(h, training=False)
            if pred_index is None:
                pred_index = int(tf.argmax(h[0]))
            class_channel = h[:, pred_index]
        grads = tape.gradient(class_channel, features)
    else:
        # ---- from-scratch CNN: standard functional Grad-CAM model ---- #
        target = _last_conv_layer(model)
        grad_model = tf.keras.models.Model(model.inputs, [target.output, model.output])
        with tf.GradientTape() as tape:
            features, preds = grad_model(img, training=False)
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, features)

    # Channel importance = mean gradient per feature-map channel.
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    features = features[0]
    heatmap = tf.squeeze(features @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_heatmap(
    image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Blend a Grad-CAM heatmap over the original image using a matplotlib
    colormap (no OpenCV dependency, so it deploys cleanly on Streamlit Cloud).

    ``image_rgb`` : uint8 array (H, W, 3). Returns a uint8 RGB array, same size.
    """
    import matplotlib.cm as cm
    from PIL import Image

    h, w = image_rgb.shape[:2]
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    heat = np.asarray(heat) / 255.0

    colored = cm.get_cmap("jet")(heat)[..., :3]        # RGBA -> RGB
    colored = np.uint8(colored * 255)
    return np.uint8(image_rgb * (1 - alpha) + colored * alpha)

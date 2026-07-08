"""
Streamlit demo for the Plant Leaf Disease Classifier.

Upload a leaf photo → get the predicted crop + disease, a confidence score, the
top-3 alternatives, and a Grad-CAM heatmap showing which part of the leaf drove
the decision.

Run
---
    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the `src` package importable whether the app is launched from the repo
# root or from inside app/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CFG                       # noqa: E402

st.set_page_config(page_title="Plant Leaf Disease Classifier",
                   page_icon="🌿", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def _load():
    """Load model + labels once and cache across reruns/sessions."""
    from src.predict import load_model_and_labels
    return load_model_and_labels()


def _confidence_row(label: str, conf: float, top: bool):
    """Render a label with an inline confidence bar."""
    pct = conf * 100
    colour = "#16a34a" if top else "#94a3b8"
    st.markdown(
        f"""
        <div style="margin:6px 0;">
          <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
            <span>{label}</span><span><b>{pct:.1f}%</b></span>
          </div>
          <div style="background:#e2e8f0;border-radius:6px;height:10px;">
            <div style="width:{pct:.1f}%;background:{colour};height:10px;border-radius:6px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("🌿 Plant Leaf Disease Classifier")
    st.caption(
        "Deep-learning model (MobileNetV2 transfer learning) trained on 38 crop-"
        "disease classes from the New Plant Diseases Dataset. Upload a leaf photo "
        "to get a diagnosis with an explainability heatmap."
    )

    # Graceful message if the trained model isn't present yet.
    if not CFG.model_path.exists():
        st.warning(
            "No trained model found at `models/plant_disease_model.keras`.\n\n"
            "Train one first with **`python -m src.train`** (or run the notebook), "
            "then reload this page."
        )
        st.stop()

    model, class_names = _load()

    uploaded = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
    show_cam = st.toggle("Show Grad-CAM explanation", value=True)

    if uploaded is None:
        st.info("👆 Upload a clear photo of a single leaf to begin.")
        return

    from PIL import Image
    from src.predict import predict

    pil_img = Image.open(uploaded).convert("RGB")

    if not st.button("🔍 Diagnose", type="primary", use_container_width=True):
        st.image(pil_img, caption="Uploaded leaf", use_container_width=True)
        return

    with st.spinner("Analysing…"):
        result = predict(pil_img, model=model, class_names=class_names, top_k=3)

    # ---- headline verdict ---- #
    if result["healthy"]:
        st.success(f"**{result['prediction']}**  ·  {result['confidence']*100:.1f}% confidence")
    else:
        st.error(f"**{result['prediction']}**  ·  {result['confidence']*100:.1f}% confidence")

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded leaf", use_container_width=True)

    with col2:
        if show_cam:
            from src.gradcam import make_gradcam_heatmap, overlay_heatmap

            heatmap, _ = make_gradcam_heatmap(result["batch"], model)
            overlay = overlay_heatmap(result["image"], heatmap, alpha=0.45)
            st.image(overlay, caption="Grad-CAM (model attention)", use_container_width=True)
        else:
            st.empty()

    st.subheader("Top predictions")
    for i, r in enumerate(result["top_k"]):
        _confidence_row(r["pretty"], r["confidence"], top=(i == 0))

    st.divider()
    st.caption(
        "⚠️ Research/educational demo — not a substitute for professional "
        "agronomic diagnosis. Model accuracy depends on image quality and is "
        "limited to the 38 trained crop-disease classes."
    )


if __name__ == "__main__":
    main()

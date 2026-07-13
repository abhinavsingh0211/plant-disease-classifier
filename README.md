# Plant Leaf Disease Classifier

A deep learning model that identifies crop disease from a photo of a leaf. Built with transfer learning on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (~87,900 images, 38 disease classes), with Grad-CAM explainability and a Streamlit demo.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Keras 3" src="https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

---

## Demo

```bash
streamlit run app/app.py
```

Upload a leaf photo and get the predicted disease, a confidence score, the top 3 alternatives, and a Grad-CAM heatmap showing where the model looked.

This needs a trained model at `models/plant_disease_model.keras` — train it first using the steps below.

---

## Results

Measured on a held-out test set of 8,800 images the model never saw during training:

| Backbone | Test accuracy | Macro-F1 | Top-3 accuracy |
|---|---:|---:|---:|
| MobileNetV2 (fine-tuned) | 98.24% | 0.9821 | 99.91% |

The training script also supports ResNet50, EfficientNetB0, and a from-scratch CNN baseline (`--backbone cnn`), but I've only fully trained and reported MobileNetV2 so far.

![Training curves](assets/accuracy_loss_graph.png)
![Confusion matrix](assets/confusion_matrix.png)

---

## How it works

```
leaf image → resize to 224×224 → augmentation (training only)
→ preprocessing → MobileNetV2 (ImageNet weights) → pooling → dropout
→ dense layer (38 classes) → prediction + Grad-CAM heatmap
```

Full pipeline: download the dataset, build a tf.data pipeline, train in two phases (frozen backbone, then fine-tune), evaluate on the held-out test set, run Grad-CAM, serve through Streamlit.

---

## Quickstart

### Option A — Google Colab (free GPU)
Open [`notebooks/plant_disease_training.ipynb`](notebooks/plant_disease_training.ipynb), set the runtime to GPU, and run all cells.

### Option B — Local
```bash
pip install -r requirements.txt

python -m src.train                     # trains MobileNetV2 by default
python -m src.evaluate                  # evaluates on the held-out test set
python -m src.tune --max-epochs 10      # optional: hyperparameter search
python -m src.predict path/to/leaf.jpg  # predict a single image
streamlit run app/app.py                # launch the demo
```

`make train`, `make evaluate`, `make app`, and `make docker` also work.

No Kaggle token needed on Colab. Running locally, put your `kaggle.json` in `~/.kaggle/`.

---

## Project structure

```
plant-disease-classifier/
├── src/
│   ├── config.py       # paths, hyperparameters, class list
│   ├── data.py         # data download and loading
│   ├── models.py       # model architectures
│   ├── train.py        # training script
│   ├── tune.py         # hyperparameter search
│   ├── evaluate.py     # test-set evaluation
│   ├── gradcam.py      # Grad-CAM
│   ├── predict.py      # single-image prediction
│   └── utils.py        # helper functions
├── app/
│   └── app.py          # Streamlit demo
├── notebooks/
│   └── plant_disease_training.ipynb
├── models/              # trained model + metrics
├── assets/              # generated plots
```

---

## A few design choices worth explaining

**Preprocessing lives inside the model.** MobileNetV2 expects pixel values in a specific range. Rather than preprocessing images before feeding them in, that step is built into the model itself, so training and prediction always run the exact same preprocessing — no risk of the two drifting apart over time.

**Test data stays separate from validation data.** Using the same data to both tune the model and report its final score gives an inflated, misleading number. The validation set is used during training; the test set is only touched once, at the end, for the real result.

**BatchNorm layers stay frozen during fine-tuning.** These layers hold statistics learned from ImageNet. Updating them on a comparatively small fine-tuning run tends to hurt more than help, so they're left as-is while the rest of the network adapts.

**Grad-CAM needed a workaround.** The pretrained backbone sits inside the model as a single nested block, so its internal layers aren't directly reachable from outside. Grad-CAM here works by manually running the model forward, grabbing the backbone's output feature map, and computing gradients from there.

---

## Limitations

Trained only on clean, lab-style photos — a single leaf, plain background — so it won't hold up as well on messy real-world photos. Limited to the 38 classes it was trained on, and it's not a substitute for actual agronomic diagnosis.

---

## License
MIT — see [LICENSE](LICENSE).

Dataset: New Plant Diseases Dataset (Kaggle, derived from PlantVillage). Model pretrained on ImageNet.

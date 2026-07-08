"""
Central configuration for the Plant Leaf Disease Classifier.

Everything that a reader might want to tweak (image size, batch size, learning
rates, paths, which backbone to use) lives here so the rest of the codebase stays
declarative. Values can be overridden from the environment, which is handy on
Colab / Kaggle where paths differ.

Usage
-----
>>> from src.config import CFG
>>> CFG.image_size
(224, 224)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# Canonical 38 classes of the "New Plant Diseases Dataset" (PlantVillage-based).
# image_dataset_from_directory() returns class names sorted alphabetically, so
# this list also encodes the exact index -> label mapping the model is trained
# with. It is used only as a fallback: after training we always persist the true
# class names to models/class_names.json and load that at inference time.
# --------------------------------------------------------------------------- #
CLASS_NAMES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Project root = parent of the `src/` directory that holds this file.
_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    # ---- Kaggle dataset ---------------------------------------------------- #
    kaggle_dataset: str = "vipoooool/new-plant-diseases-dataset"

    # ---- Image / batching -------------------------------------------------- #
    image_size: tuple[int, int] = (224, 224)   # MobileNetV2 / ResNet50 native size
    batch_size: int = 32
    seed: int = 42

    # ---- Training ---------------------------------------------------------- #
    # options: mobilenetv2 | resnet50 | efficientnetb0 | cnn
    backbone: str = _env("BACKBONE", "mobilenetv2")
    dropout: float = 0.3
    # Phase 1: train only the new head with the backbone frozen.
    head_epochs: int = 8
    head_lr: float = 1e-3
    # Phase 2: unfreeze the top of the backbone and fine-tune with a tiny LR.
    fine_tune: bool = True
    fine_tune_epochs: int = 8
    fine_tune_lr: float = 1e-5
    fine_tune_at: float = 0.7     # unfreeze the last 30% of backbone layers
    early_stop_patience: int = 4

    # ---- Paths ------------------------------------------------------------- #
    root: Path = _ROOT
    models_dir: Path = _ROOT / "models"
    assets_dir: Path = _ROOT / "assets"

    # Filled in at runtime once the dataset is located on disk.
    data_dir: Path | None = None

    # ---- Derived artefact paths ------------------------------------------- #
    @property
    def model_path(self) -> Path:
        return self.models_dir / "plant_disease_model.keras"

    @property
    def class_names_path(self) -> Path:
        return self.models_dir / "class_names.json"

    @property
    def metrics_path(self) -> Path:
        return self.models_dir / "metrics.json"

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


# A ready-to-import singleton.
CFG = Config()

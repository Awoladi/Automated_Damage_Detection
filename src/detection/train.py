"""
BIMInspect — YOLOv8 Classification Training
Fine-tunes yolov8n-cls on the crack / no-crack dataset.

Dataset layout expected:
    data/raw/Positive/   — crack images
    data/raw/Negative/   — no-crack images

What this script does:
    1. Splits images into train / val / test sets (70 / 20 / 10 %)
    2. Writes the split folders under data/splits/
    3. Saves a dataset config to models/configs/dataset.yaml
    4. Trains yolov8n-cls for N epochs on the GPU
    5. Saves the best weights to models/weights/
"""

import os
import random
import shutil
import yaml
from pathlib import Path

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]   # repo root
RAW_POS     = ROOT / "data" / "raw" / "Positive"
RAW_NEG     = ROOT / "data" / "raw" / "Negative"
SPLITS_DIR  = ROOT / "data" / "splits"
CONFIGS_DIR = ROOT / "models" / "configs"
WEIGHTS_DIR = ROOT / "models" / "weights"

CLASSES = {"crack": RAW_POS, "no_crack": RAW_NEG}

# ── Hyper-parameters ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
# TEST_RATIO = 0.10  (remainder)

EPOCHS      = 20
IMG_SIZE    = 224
BATCH       = 32
BASE_MODEL  = "yolov8n-cls.pt"   # nano classification checkpoint
SEED        = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def split_class(class_name: str, src_dir: Path) -> dict[str, int]:
    """Copy images from src_dir into train/val/test/<class_name> folders."""
    images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No images found in {src_dir}")

    random.seed(SEED)
    random.shuffle(images)

    n        = len(images)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    counts = {}
    for split_name, imgs in splits.items():
        dest = SPLITS_DIR / split_name / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dest / img.name)
        counts[split_name] = len(imgs)
        print(f"  {split_name:5s}  {class_name:8s}  {len(imgs):,} images → {dest}")

    return counts


def prepare_splits() -> bool:
    """Return True if splits already exist and are populated, False otherwise."""
    train_crack = SPLITS_DIR / "train" / "crack"
    if train_crack.exists() and any(train_crack.iterdir()):
        print("Splits already exist — skipping copy step.")
        return True
    return False


def write_dataset_yaml() -> Path:
    """Write a dataset.yaml that documents the split layout."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "task":    "classify",
        "path":    str(SPLITS_DIR),
        "train":   "train",
        "val":     "val",
        "test":    "test",
        "nc":      len(CLASSES),
        "names":   list(CLASSES.keys()),
    }
    yaml_path = CONFIGS_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Dataset config written → {yaml_path}")
    return yaml_path


def train() -> None:
    # ── 1. Build splits ────────────────────────────────────────────────────────
    if not prepare_splits():
        print("Building train / val / test splits …")
        for class_name, src_dir in CLASSES.items():
            split_class(class_name, src_dir)
        print()

    # ── 2. Dataset YAML ────────────────────────────────────────────────────────
    write_dataset_yaml()
    print()

    # ── 3. Train ───────────────────────────────────────────────────────────────
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"Starting training — {EPOCHS} epochs, img {IMG_SIZE}px, batch {BATCH}")
    print(f"Dataset: {SPLITS_DIR}\n")

    results = model.train(
        data    = str(SPLITS_DIR),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        device  = 0,                      # GPU 0 (RTX 3070)
        project = str(WEIGHTS_DIR),
        name    = "biminspect-cls",
        exist_ok= True,
        patience= 5,                      # early stopping
        augment = True,
        verbose = True,
    )

    # ── 4. Copy best weights to a fixed path ──────────────────────────────────
    run_dir   = WEIGHTS_DIR / "biminspect-cls"
    best_src  = run_dir / "weights" / "best.pt"
    best_dest = WEIGHTS_DIR / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dest)
        print(f"\nBest weights saved → {best_dest}")
    else:
        print(f"\nWarning: best.pt not found at {best_src}")

    print("\nTraining complete.")
    print(f"  Top-1 accuracy: {results.results_dict.get('metrics/accuracy_top1', 'n/a')}")
    print(f"  Top-5 accuracy: {results.results_dict.get('metrics/accuracy_top5', 'n/a')}")


if __name__ == "__main__":
    train()

"""
BIMInspect — YOLOv8 Object Detection Training
Fine-tunes yolov8n.pt on the auto-labeled crack detection dataset.

Requires data/detection/ to be built first:
    python src/detection/generate_labels.py

Saves best weights to models/weights/best_detection.pt
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DET_DIR     = ROOT / "data" / "detection"
CONFIGS_DIR = ROOT / "models" / "configs"
WEIGHTS_DIR = ROOT / "models" / "weights"

BASE_MODEL  = "yolov8n.pt"
EPOCHS      = 50
IMG_SIZE    = 640          # standard YOLO detection size (upscaled from 227px)
BATCH       = 16
PATIENCE    = 10
SEED        = 42


def write_dataset_yaml() -> Path:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "path":  str(DET_DIR),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["crack"],
    }
    out = CONFIGS_DIR / "dataset_detection.yaml"
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Dataset YAML written: {out}")
    return out


def train() -> None:
    if not (DET_DIR / "train" / "images").exists():
        raise RuntimeError(
            "Detection dataset not found.\n"
            "Run:  python src/detection/generate_labels.py"
        )

    yaml_path = write_dataset_yaml()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"Training YOLOv8n detection — {EPOCHS} epochs, img {IMG_SIZE}px, batch {BATCH}\n")
    results = model.train(
        data     = str(yaml_path),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        device   = 0,
        project  = str(WEIGHTS_DIR),
        name     = "biminspect-det",
        exist_ok = True,
        patience = PATIENCE,
        seed     = SEED,
        verbose  = True,
        # Augmentation — helps given the Grad-CAM label noise
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        flipud   = 0.3,
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.1,
    )

    best_src  = WEIGHTS_DIR / "biminspect-det" / "weights" / "best.pt"
    best_dest = WEIGHTS_DIR / "best_detection.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dest)
        print(f"\nBest weights saved: {best_dest}")

    metrics = results.results_dict
    print("\nTraining complete.")
    print(f"  mAP50   : {metrics.get('metrics/mAP50(B)',   'n/a')}")
    print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)','n/a')}")
    print(f"  Precision: {metrics.get('metrics/precision(B)','n/a')}")
    print(f"  Recall   : {metrics.get('metrics/recall(B)',   'n/a')}")


if __name__ == "__main__":
    train()

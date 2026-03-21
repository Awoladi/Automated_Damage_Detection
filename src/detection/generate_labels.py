"""
BIMInspect — YOLO Detection Label Generator
Converts the classification dataset into a YOLO object-detection dataset by using
Grad-CAM activations from the trained classifier to auto-generate bounding boxes.

Output layout (YOLO detection format):
    data/detection/
        train/
            images/   *.jpg
            labels/   *.txt   (one line: "0 cx cy w h" per crack)
        val/
            images/   *.jpg
            labels/   *.txt
        test/
            images/   *.jpg
            labels/   *.txt

Label format per file:
    <class_id> <cx> <cy> <w> <h>   — all values normalised 0-1, YOLO standard
    class 0 = crack

No-crack images are included as background examples with empty label files,
which teaches the detector to suppress false positives.

Grad-CAM improvements over the classifier detector:
  - Lower heatmap threshold (0.30) captures the full crack extent
  - Morphological dilation expands the mask slightly before bbox fitting
  - Minimum area filter discards noise activations (< 1% of image area)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
RAW_POS      = ROOT / "data" / "raw" / "Positive"
RAW_NEG      = ROOT / "data" / "raw" / "Negative"
DET_DIR      = ROOT / "data" / "detection"
WEIGHTS      = ROOT / "models" / "weights" / "best.pt"

# ── Split ratios (must match training split) ───────────────────────────────────
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20
SEED         = 42

# ── Grad-CAM settings ──────────────────────────────────────────────────────────
HEATMAP_THRESHOLD  = 0.30          # lower = larger, more conservative boxes
DILATE_ITERATIONS  = 3             # expand mask to capture full crack extent
MIN_AREA_FRACTION  = 0.005         # discard components < 0.5% of image area
CRACK_CLASS_ID     = 0
IMG_SIZE           = 224


# ── Grad-CAM (inline — no dep on detector.py to keep this script self-contained) ──

class _GradCAM:
    def __init__(self, inner: torch.nn.Module) -> None:
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        target = inner.model[9].conv.conv
        self._fh = target.register_forward_hook(
            lambda _m, _i, o: setattr(self, '_acts', o.detach())
        )
        self._bh = target.register_full_backward_hook(
            lambda _m, _gi, go: setattr(self, '_grads', go[0].detach())
        )

    def compute(self, logits: torch.Tensor, cls: int, hw: tuple[int, int]) -> np.ndarray:
        logits[0, cls].backward(retain_graph=True)
        w   = self._grads[0].mean(dim=(1, 2))
        cam = F.relu(torch.einsum('c,chw->hw', w, self._acts[0]))
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
        return (cam / cam.max()).astype(np.float32) if cam.max() > 0 else cam

    def remove(self) -> None:
        self._fh.remove()
        self._bh.remove()


def _gradcam_yolo_label(
    inner: torch.nn.Module,
    img_bgr: np.ndarray,
    device: str,
) -> str | None:
    """
    Run Grad-CAM and return a YOLO label line "0 cx cy w h" or None if
    no valid crack region is found.
    """
    h, w = img_bgr.shape[:2]
    inner.train()

    # disable inplace ops
    saved = []
    for m in inner.modules():
        if hasattr(m, 'inplace'):
            saved.append((m, m.inplace))
            m.inplace = False

    gc = _GradCAM(inner)
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = (
            torch.tensor(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)), dtype=torch.float32)
            .permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        )
        t.requires_grad_(True)
        out    = inner(t)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        heatmap = gc.compute(logits, CRACK_CLASS_ID, hw=(h, w))
    finally:
        gc.remove()
        inner.eval()
        for m, s in saved:
            m.inplace = s

    # threshold + dilate
    binary = (heatmap >= HEATMAP_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=DILATE_ITERATIONS)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return None

    min_area = MIN_AREA_FRACTION * h * w
    valid = [
        i for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if not valid:
        return None

    # Union box over all valid components
    x1 = min(stats[i, cv2.CC_STAT_LEFT]  for i in valid)
    y1 = min(stats[i, cv2.CC_STAT_TOP]   for i in valid)
    x2 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  for i in valid)
    y2 = max(stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] for i in valid)

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return f"{CRACK_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_detection_dataset(force: bool = False) -> None:
    # Check if already built
    check = DET_DIR / "train" / "images"
    if check.exists() and any(check.iterdir()) and not force:
        print("Detection dataset already exists — skipping. Pass force=True to rebuild.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading classifier weights from {WEIGHTS} (device={device})")
    model_wrap = YOLO(str(WEIGHTS))
    inner      = model_wrap.model.to(device)
    inner.eval()

    import random
    random.seed(SEED)

    pos_imgs = sorted(RAW_POS.glob("*.jpg"))
    neg_imgs = sorted(RAW_NEG.glob("*.jpg"))
    random.shuffle(pos_imgs)
    random.shuffle(neg_imgs)

    def split(imgs):
        n       = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        return (
            imgs[:n_train],
            imgs[n_train:n_train + n_val],
            imgs[n_train + n_val:],
        )

    pos_train, pos_val, pos_test = split(pos_imgs)
    neg_train, neg_val, neg_test = split(neg_imgs)

    splits = {
        "train": (pos_train, neg_train),
        "val":   (pos_val,   neg_val),
        "test":  (pos_test,  neg_test),
    }

    total_imgs    = len(pos_imgs) + len(neg_imgs)
    total_labeled = 0
    total_skipped = 0
    processed     = 0
    t0            = time.time()

    print(f"Generating YOLO labels for {total_imgs:,} images …\n")

    for split_name, (pos, neg) in splits.items():
        img_dir = DET_DIR / split_name / "images"
        lbl_dir = DET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # ── Positive (crack) images ────────────────────────────────────────────
        for img_path in pos:
            dest_img = img_dir / img_path.name
            dest_lbl = lbl_dir / img_path.with_suffix(".txt").name

            shutil.copy2(img_path, dest_img)
            img_bgr = cv2.imread(str(img_path))

            label = _gradcam_yolo_label(inner, img_bgr, device)
            if label:
                dest_lbl.write_text(label)
                total_labeled += 1
            else:
                dest_lbl.write_text("")   # no valid region — treat as background
                total_skipped += 1

            processed += 1
            if processed % 500 == 0:
                elapsed = time.time() - t0
                rate    = processed / elapsed
                remain  = (total_imgs - processed) / rate
                print(
                    f"  {processed:>6}/{total_imgs}  "
                    f"labeled={total_labeled}  skipped={total_skipped}  "
                    f"eta={remain/60:.1f}min"
                )

        # ── Negative (no-crack) images — empty labels ──────────────────────────
        for img_path in neg:
            shutil.copy2(img_path, img_dir / img_path.name)
            (lbl_dir / img_path.with_suffix(".txt").name).write_text("")
            processed += 1

        print(
            f"  [{split_name}] {len(pos)} crack + {len(neg)} no-crack images written"
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Crack images labeled : {total_labeled:,}")
    print(f"  Skipped (no region)  : {total_skipped:,}")
    print(f"  Background images    : {len(neg_imgs):,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate YOLO detection labels via Grad-CAM")
    p.add_argument("--force", action="store_true", help="Rebuild even if dataset exists")
    args = p.parse_args()
    build_detection_dataset(force=args.force)

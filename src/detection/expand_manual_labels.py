"""
BIMInspect — Manual Label Expander
Multiplies the 500 hand-labelled images using exact geometric transforms:
  - Original
  - Horizontal flip
  - Vertical flip
  - Rotate 90°
  - Rotate 180°
  - Rotate 270°
  - Transpose (flip diagonal)
  - Transverse (flip anti-diagonal)

Each of the 8 transforms produces pixel-perfect bbox coordinates — no
interpolation, no coordinate rounding errors. 500 × 8 = 4,000 images.

Output: data/expanded_manual/
  train/images/  train/labels/   (~3400 images, 85%)
  val/images/    val/labels/     (~600 images,  15%)
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT       = Path(__file__).resolve().parents[2]
ANNOTATED  = ROOT / "data" / "annotated" / "labels"
SAMPLE_DIR = ROOT / "data" / "labeling" / "sample_500"
OUT_DIR    = ROOT / "data" / "expanded_manual"
VAL_FRAC   = 0.15
SEED       = 42

random.seed(SEED)


# ── YOLO helpers ───────────────────────────────────────────────────────────────

def parse_boxes(txt: str) -> list[tuple]:
    rows = []
    for line in txt.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
    return rows


def fmt(boxes: list[tuple]) -> str:
    return "\n".join(
        f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, cx, cy, w, h in boxes
    )


# ── 8 exact transforms ─────────────────────────────────────────────────────────

def original(img, boxes):
    return img, boxes

def flip_h(img, boxes):
    return cv2.flip(img, 1), [(c, 1-cx, cy, w, h) for c, cx, cy, w, h in boxes]

def flip_v(img, boxes):
    return cv2.flip(img, 0), [(c, cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]

def rot90(img, boxes):
    # 90° CCW: (cx,cy) -> (cy, 1-cx), w<->h
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), \
           [(c, cy, 1-cx, h, w) for c, cx, cy, w, h in boxes]

def rot180(img, boxes):
    return cv2.rotate(img, cv2.ROTATE_180), \
           [(c, 1-cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]

def rot270(img, boxes):
    # 90° CW: (cx,cy) -> (1-cy, cx), w<->h
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), \
           [(c, 1-cy, cx, h, w) for c, cx, cy, w, h in boxes]

def transpose(img, boxes):
    # Flip along main diagonal: (cx,cy) -> (cy,cx), w<->h
    return cv2.transpose(img), \
           [(c, cy, cx, h, w) for c, cx, cy, w, h in boxes]

def transverse(img, boxes):
    # Flip along anti-diagonal = rot90 + transpose
    img_t, boxes_t = rot90(img, boxes)
    return transpose(img_t, boxes_t)


TRANSFORMS = [
    ("orig",       original),
    ("fliph",      flip_h),
    ("flipv",      flip_v),
    ("rot90",      rot90),
    ("rot180",     rot180),
    ("rot270",     rot270),
    ("transpose",  transpose),
    ("transverse", transverse),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def expand() -> None:
    label_files = sorted(ANNOTATED.glob("*.txt"))

    all_samples: list[tuple[str, np.ndarray, list]] = []   # (name, img, boxes)

    for lbl_path in label_files:
        stem     = lbl_path.stem.split("__", 1)[-1]
        img_path = SAMPLE_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue
        img   = cv2.imread(str(img_path))
        boxes = parse_boxes(lbl_path.read_text())
        if img is None or not boxes:
            continue

        for t_name, t_fn in TRANSFORMS:
            t_img, t_boxes = t_fn(img, boxes)
            all_samples.append((f"{stem}__{t_name}", t_img, t_boxes))

    random.shuffle(all_samples)
    n_val   = int(len(all_samples) * VAL_FRAC)
    splits  = {"val": all_samples[:n_val], "train": all_samples[n_val:]}

    print(f"Source images   : {len(label_files)}")
    print(f"Transforms      : {len(TRANSFORMS)} per image")
    print(f"Total samples   : {len(all_samples)}")

    for split_name, samples in splits.items():
        img_dir = OUT_DIR / split_name / "images"
        lbl_dir = OUT_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for name, img, boxes in samples:
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            (lbl_dir / f"{name}.txt").write_text(fmt(boxes))

        print(f"  [{split_name}] {len(samples)} images written -> {img_dir}")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    expand()
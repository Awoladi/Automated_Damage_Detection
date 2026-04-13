"""
BIMInspect — Multi-class Label Expander

Applies 8-way geometric augmentation to all classes in
data/annotated_multiclass/ and combines with existing crack data
from data/expanded_manual/.

Output: data/expanded_multiclass/
  train/images/ + train/labels/
  val/images/   + val/labels/

Classes:
  0: crack        (from data/expanded_manual/ — our high-quality manual labels)
  1: spallation   (from CODEBRIM)
  2: efflorescence(from CODEBRIM)
  3: exposed_bars (from CODEBRIM)
  4: corrosion    (from CODEBRIM)
"""

from __future__ import annotations
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT         = Path(__file__).resolve().parents[2]
MULTICLASS   = ROOT / "data" / "annotated_multiclass"
CRACK_DATA   = ROOT / "data" / "expanded_manual"
OUT_DIR      = ROOT / "data" / "expanded_multiclass"
VAL_FRAC     = 0.15
SEED         = 42

random.seed(SEED)

CLASSES = ["crack", "spallation", "efflorescence", "exposed_bars", "corrosion"]


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

def original(img, boxes):   return img, boxes
def flip_h(img, boxes):     return cv2.flip(img, 1), [(c, 1-cx, cy, w, h) for c, cx, cy, w, h in boxes]
def flip_v(img, boxes):     return cv2.flip(img, 0), [(c, cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]
def rot90(img, boxes):      return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), [(c, cy, 1-cx, h, w) for c, cx, cy, w, h in boxes]
def rot180(img, boxes):     return cv2.rotate(img, cv2.ROTATE_180), [(c, 1-cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]
def rot270(img, boxes):     return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), [(c, 1-cy, cx, h, w) for c, cx, cy, w, h in boxes]
def transpose(img, boxes):  return cv2.transpose(img), [(c, cy, cx, h, w) for c, cx, cy, w, h in boxes]
def transverse(img, boxes):
    img_t, boxes_t = rot90(img, boxes)
    return transpose(img_t, boxes_t)

TRANSFORMS = [
    ("orig", original), ("fliph", flip_h), ("flipv", flip_v),
    ("rot90", rot90), ("rot180", rot180), ("rot270", rot270),
    ("transpose", transpose), ("transverse", transverse),
]


# ── Main ───────────────────────────────────────────────────────────────────────

IMG_SIZE = 640  # resize all images to this before augmenting — saves RAM


def expand_class_to_disk(cls_name: str, cls_id: int,
                          train_img: Path, train_lbl: Path,
                          val_img: Path,   val_lbl: Path) -> tuple[int, int]:
    """Process one class at a time, writing directly to disk. Never holds all images in RAM."""
    img_dir = MULTICLASS / cls_name / "images"
    lbl_dir = MULTICLASS / cls_name / "labels"
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    random.shuffle(img_paths)
    n_val = max(1, int(len(img_paths) * VAL_FRAC))
    val_set   = set(p.stem for p in img_paths[:n_val])

    written_train = written_val = 0

    for img_path in img_paths:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Resize to fixed size to keep RAM usage bounded
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        boxes = parse_boxes(lbl_path.read_text())
        if not boxes:
            continue
        boxes = [(cls_id, cx, cy, w, h) for _, cx, cy, w, h in boxes]

        is_val = img_path.stem in val_set
        out_img = val_img   if is_val else train_img
        out_lbl = val_lbl   if is_val else train_lbl

        for t_name, t_fn in TRANSFORMS:
            t_img, t_boxes = t_fn(img, boxes)
            name = f"{cls_name}__{img_path.stem}__{t_name}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), t_img)
            (out_lbl / f"{name}.txt").write_text(fmt(t_boxes))
            if is_val: written_val += 1
            else:      written_train += 1

        del img  # free memory immediately

    return written_train, written_val


def copy_crack_data() -> tuple[int, int]:
    """Copy existing expanded crack data into the new combined dataset."""
    train_src = CRACK_DATA / "train"
    val_src   = CRACK_DATA / "val"
    copied_train = copied_val = 0

    for split, src in [("train", train_src), ("val", val_src)]:
        img_out = OUT_DIR / split / "images"
        lbl_out = OUT_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in (src / "images").glob("*.jpg"):
            shutil.copy2(img, img_out / img.name)
        for lbl in (src / "labels").glob("*.txt"):
            shutil.copy2(lbl, lbl_out / lbl.name)

        count = len(list((src / "images").glob("*.jpg")))
        if split == "train": copied_train = count
        else: copied_val = count

    return copied_train, copied_val


def main() -> None:
    # Create output dirs once
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    train_img = OUT_DIR / "train" / "images"
    train_lbl = OUT_DIR / "train" / "labels"
    val_img   = OUT_DIR / "val"   / "images"
    val_lbl   = OUT_DIR / "val"   / "labels"

    total_train = total_val = 0

    print("Expanding CODEBRIM classes (writing directly to disk) ...")
    for cls_id, cls_name in enumerate(CLASSES):
        if cls_name == "crack":
            continue
        wt, wv = expand_class_to_disk(cls_name, cls_id,
                                       train_img, train_lbl,
                                       val_img,   val_lbl)
        print(f"  {cls_name:20s}: {wt} train, {wv} val")
        total_train += wt
        total_val   += wv

    print("\nCopying existing crack data ...")
    crack_train, crack_val = copy_crack_data()
    print(f"  crack:                {crack_train} train, {crack_val} val")

    total_train += crack_train
    total_val   += crack_val
    print(f"\nFinal dataset: {total_train} train, {total_val} val")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

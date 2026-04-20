"""
BIMInspect — CODEBRIM Bbox Dataset Builder

Reads CODEBRIM_original_images.zip, parses per-image XML bounding box annotations,
applies 8-way geometric augmentation, and writes the full multi-class training
dataset to data/expanded_multiclass/.

Also copies existing crack data from data/expanded_manual/.

Input:
  data/raw/CODEBRIM_original_images.zip   (7.8 GB, already downloaded)
  data/expanded_manual/                   (crack data from expand_manual_labels.py)

Output:
  data/expanded_multiclass/
    train/images/ + train/labels/
    val/images/   + val/labels/

Classes:
  0: crack         (from data/expanded_manual/ — high-quality manual labels)
  1: spallation    (CODEBRIM original images, real bboxes)
  2: efflorescence (CODEBRIM original images, real bboxes)
  3: exposed_bars  (CODEBRIM original images, real bboxes)
  4: corrosion     (CODEBRIM original images, real bboxes)
"""

from __future__ import annotations
import random
import shutil
import struct
import zipfile
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT      = Path(__file__).resolve().parents[2]
ZIP_PATH  = ROOT / "data" / "raw" / "CODEBRIM_original_images.zip"
CRACK_SRC = ROOT / "data" / "expanded_manual"
OUT_DIR   = ROOT / "data" / "expanded_multiclass"
VAL_FRAC  = 0.15
SEED      = 42
IMG_SIZE  = 640
ZIP_SHIFT = 4_294_967_296

random.seed(SEED)

# CODEBRIM XML tag → YOLO class id (Crack skipped — better data from manual labels)
TAG_TO_CLS = {
    "Spallation":     1,
    "Efflorescence":  2,
    "ExposedBars":    3,
    "CorrosionStain": 4,
}


# ── ZIP reader (handles 4 GB offset corruption) ────────────────────────────────

def read_zip_entry(zip_path: Path, info: zipfile.ZipInfo) -> bytes:
    with open(zip_path, "rb") as f:
        for offset in [info.header_offset, info.header_offset - ZIP_SHIFT]:
            f.seek(offset)
            if f.read(4) == b"PK\x03\x04":
                f.seek(offset + 26)
                fname_len = struct.unpack("<H", f.read(2))[0]
                extra_len = struct.unpack("<H", f.read(2))[0]
                f.seek(fname_len + extra_len, 1)
                raw = f.read(info.compress_size)
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    return zlib.decompress(raw, -15)
                if info.compress_type == zipfile.ZIP_STORED:
                    return raw
                raise ValueError(f"Unsupported compression: {info.compress_type}")
    raise ValueError(f"PK magic not found for: {info.filename}")


# ── YOLO format helpers ────────────────────────────────────────────────────────

def boxes_to_yolo(boxes: list[tuple], img_w: int, img_h: int) -> list[tuple]:
    """Convert (cls, xmin, ymin, xmax, ymax) pixel coords to YOLO normalised format."""
    yolo = []
    for cls, x1, y1, x2, y2 in boxes:
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        yolo.append((cls, cx, cy, w, h))
    return yolo


def fmt(boxes: list[tuple]) -> str:
    return "\n".join(
        f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, cx, cy, w, h in boxes
    )


# ── 8-way geometric augmentation ──────────────────────────────────────────────

def original(img, boxes):
    return img, boxes

def flip_h(img, boxes):
    return cv2.flip(img, 1), [(c, 1-cx, cy, w, h) for c, cx, cy, w, h in boxes]

def flip_v(img, boxes):
    return cv2.flip(img, 0), [(c, cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]

def rot90(img, boxes):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), \
           [(c, cy, 1-cx, h, w) for c, cx, cy, w, h in boxes]

def rot180(img, boxes):
    return cv2.rotate(img, cv2.ROTATE_180), \
           [(c, 1-cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]

def rot270(img, boxes):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), \
           [(c, 1-cy, cx, h, w) for c, cx, cy, w, h in boxes]

def transpose(img, boxes):
    return cv2.transpose(img), [(c, cy, cx, h, w) for c, cx, cy, w, h in boxes]

def transverse(img, boxes):
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


# ── XML parser ────────────────────────────────────────────────────────────────

def parse_xml(data: bytes) -> tuple[int, int, list[tuple]]:
    """
    Returns (img_w, img_h, boxes) where boxes = [(cls_id, xmin, ymin, xmax, ymax), ...].
    Multi-label objects produce one entry per active damage class.
    Crack and Background are skipped.
    """
    root = ET.fromstring(data)
    img_w = int(root.findtext("size/width",  0))
    img_h = int(root.findtext("size/height", 0))
    boxes = []
    for obj in root.findall("object"):
        defect = obj.find("Defect")
        if defect is None:
            continue
        xmin = int(obj.findtext("bndbox/xmin", 0))
        ymin = int(obj.findtext("bndbox/ymin", 0))
        xmax = int(obj.findtext("bndbox/xmax", 0))
        ymax = int(obj.findtext("bndbox/ymax", 0))
        for tag, cls_id in TAG_TO_CLS.items():
            if defect.findtext(tag, "0").strip() == "1":
                boxes.append((cls_id, xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"CODEBRIM original images zip not found: {ZIP_PATH}\n"
            "Download from: https://zenodo.org/records/2620293"
        )
    if not CRACK_SRC.exists():
        raise FileNotFoundError(
            f"Crack data not found: {CRACK_SRC}\n"
            "Run: python src/detection/expand_manual_labels.py"
        )

    # Clear and recreate output directories
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True)
        (OUT_DIR / split / "labels").mkdir(parents=True)

    train_img = OUT_DIR / "train" / "images"
    train_lbl = OUT_DIR / "train" / "labels"
    val_img   = OUT_DIR / "val"   / "images"
    val_lbl   = OUT_DIR / "val"   / "labels"

    # ── Step 1: CODEBRIM original images with proper bboxes ───────────────────
    print("Reading CODEBRIM zip index ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        all_entries = zf.infolist()
        xmls = {
            Path(e.filename).stem: e
            for e in all_entries
            if e.filename.endswith(".xml") and "__MACOSX" not in e.filename
        }
        imgs = {
            Path(e.filename).stem: e
            for e in all_entries
            if e.filename.endswith(".jpg") and "__MACOSX" not in e.filename
        }

    print(f"  XML annotations : {len(xmls)}")
    print(f"  JPG images      : {len(imgs)}")

    # Only process images that have both XML and JPG
    paired = sorted(set(xmls) & set(imgs))
    print(f"  Paired (xml+jpg): {len(paired)}")

    random.shuffle(paired)
    n_val   = max(1, int(len(paired) * VAL_FRAC))
    val_set = set(paired[:n_val])

    written_train = written_val = skipped = 0

    print("\nProcessing CODEBRIM images ...")
    for stem in paired:
        # Parse XML
        try:
            xml_data       = read_zip_entry(ZIP_PATH, xmls[stem])
            img_w, img_h, raw_boxes = parse_xml(xml_data)
        except Exception as e:
            skipped += 1
            continue

        if not raw_boxes:
            skipped += 1
            continue

        # Convert pixel bboxes → YOLO normalised (using original image dimensions)
        yolo_boxes = boxes_to_yolo(raw_boxes, img_w, img_h)
        if not yolo_boxes:
            skipped += 1
            continue

        # Read image
        try:
            img_data = read_zip_entry(ZIP_PATH, imgs[stem])
            img_arr  = np.frombuffer(img_data, np.uint8)
            img      = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception:
            skipped += 1
            continue

        if img is None:
            skipped += 1
            continue

        # Resize to 640×640 (bbox coords are already normalised — no adjustment needed)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        is_val  = stem in val_set
        out_img = val_img   if is_val else train_img
        out_lbl = val_lbl   if is_val else train_lbl

        for t_name, t_fn in TRANSFORMS:
            t_img, t_boxes = t_fn(img, yolo_boxes)
            name = f"codebrim__{stem}__{t_name}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), t_img)
            (out_lbl / f"{name}.txt").write_text(fmt(t_boxes))
            if is_val:
                written_val   += 1
            else:
                written_train += 1

        del img

    print(f"  Written — train: {written_train}  val: {written_val}  skipped: {skipped}")

    # ── Step 2: Copy existing crack data ──────────────────────────────────────
    print("\nCopying crack data from data/expanded_manual/ ...")
    crack_train = crack_val = 0
    for split in ("train", "val"):
        src_img = CRACK_SRC / split / "images"
        src_lbl = CRACK_SRC / split / "labels"
        dst_img = OUT_DIR   / split / "images"
        dst_lbl = OUT_DIR   / split / "labels"
        for f in src_img.glob("*.jpg"):
            shutil.copy2(f, dst_img / f.name)
        for f in src_lbl.glob("*.txt"):
            shutil.copy2(f, dst_lbl / f.name)
        n = len(list(src_img.glob("*.jpg")))
        if split == "train": crack_train = n
        else:                crack_val   = n

    print(f"  crack — train: {crack_train}  val: {crack_val}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_train = written_train + crack_train
    total_val   = written_val   + crack_val
    print(f"\nFinal dataset : {total_train} train  {total_val} val")
    print(f"Output        : {OUT_DIR}")


if __name__ == "__main__":
    main()

"""
BIMInspect — CODEBRIM Tiled Bbox Dataset Builder (v11 — domain-harmonised)

1. Tiles 640×640 crops from original full-resolution CODEBRIM images,
   including Crack annotations so all 5 classes share the same visual domain.
2. Copies supplementary crack data from data/expanded_manual/ (Kaggle)
   to add macro-level crack diversity.
3. Oversamples rare classes (repeat-factor) so each class reaches a target
   count — raises the floor rather than removing majority data.

Input:
  data/raw/CODEBRIM_original_images.zip
  data/expanded_manual/

Output:
  data/expanded_multiclass/
    train/images/ + train/labels/
    val/images/   + val/labels/

Classes:
  0: crack         (CODEBRIM tiles + Kaggle)
  1: spallation    (CODEBRIM tiles)
  2: efflorescence (CODEBRIM tiles)
  3: exposed_bars  (CODEBRIM tiles)
  4: corrosion     (CODEBRIM tiles)
"""

from __future__ import annotations
import random
import shutil
import struct
import zipfile
import zlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT      = Path(__file__).resolve().parents[2]
ZIP_PATH  = ROOT / "data" / "raw" / "CODEBRIM_original_images.zip"
CRACK_SRC = ROOT / "data" / "expanded_manual"
OUT_DIR   = ROOT / "data" / "expanded_multiclass"
VAL_FRAC  = 0.15
SEED      = 42
ZIP_SHIFT = 4_294_967_296

TILE_SIZE      = 640
TILE_STRIDE    = 480
MIN_VISIBILITY = 0.25

random.seed(SEED)

TAG_TO_CLS = {
    "Crack":          0,   # building-facade cracks — same domain as other classes
    "Spallation":     1,
    "Efflorescence":  2,
    "ExposedBars":    3,
    "CorrosionStain": 4,
}

CLS_NAMES = {0: "crack", 1: "spallation", 2: "efflorescence", 3: "exposed_bars", 4: "corrosion"}


# ── ZIP reader ────────────────────────────────────────────────────────────────

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


# ── XML parser ────────────────────────────────────────────────────────────────

def parse_xml(data: bytes) -> tuple[int, int, list[tuple]]:
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
        if xmax <= xmin or ymax <= ymin:
            continue
        for tag, cls_id in TAG_TO_CLS.items():
            if defect.findtext(tag, "0").strip() == "1":
                boxes.append((cls_id, xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes


# ── Tiling ────────────────────────────────────────────────────────────────────

def generate_tiles(img: np.ndarray, abs_boxes: list[tuple]) -> list[tuple]:
    h, w = img.shape[:2]

    def anchors(total: int) -> list[int]:
        pts = list(range(0, total - TILE_SIZE, TILE_STRIDE))
        pts.append(max(0, total - TILE_SIZE))
        return sorted(set(pts))

    results = []
    for y0 in anchors(h):
        y1 = y0 + TILE_SIZE
        for x0 in anchors(w):
            x1 = x0 + TILE_SIZE
            tile_boxes = []
            for cls, bx1, by1, bx2, by2 in abs_boxes:
                ix1 = max(bx1, x0);  ix2 = min(bx2, x1)
                iy1 = max(by1, y0);  iy2 = min(by2, y1)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                orig_area  = max(1.0, float((bx2 - bx1) * (by2 - by1)))
                if (ix2 - ix1) * (iy2 - iy1) / orig_area < MIN_VISIBILITY:
                    continue
                cx = ((ix1 + ix2) / 2 - x0) / TILE_SIZE
                cy = ((iy1 + iy2) / 2 - y0) / TILE_SIZE
                bw = (ix2 - ix1) / TILE_SIZE
                bh = (iy2 - iy1) / TILE_SIZE
                tile_boxes.append((cls, cx, cy, bw, bh))
            if tile_boxes:
                results.append((img[y0:y1, x0:x1].copy(), tile_boxes))
    return results


# ── Label I/O ─────────────────────────────────────────────────────────────────

def fmt(boxes: list[tuple]) -> str:
    return "\n".join(
        f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, cx, cy, w, h in boxes
    )


# ── Repeat-factor oversampling ────────────────────────────────────────────────

def oversample_rare_classes(
    train_img: Path,
    train_lbl: Path,
    target: int,
) -> int:
    """
    For each class below target, copy existing tiles (round-robin) until the
    class reaches target count. Returns number of tiles added.
    """
    class_to_stems: dict[int, list[str]] = defaultdict(list)
    for txt in train_lbl.glob("*.txt"):
        classes_present = {int(line.split()[0]) for line in txt.read_text().splitlines() if line.strip()}
        for cls in classes_present:
            class_to_stems[cls].append(txt.stem)

    added = 0
    for cls in sorted(class_to_stems):
        stems = class_to_stems[cls]
        needed = target - len(stems)
        if needed <= 0:
            continue
        pool = stems.copy()
        random.shuffle(pool)
        for i in range(needed):
            src_stem = pool[i % len(pool)]
            dst_stem  = f"{src_stem}__os{cls}_{i:04d}"
            src_img_f = train_img / f"{src_stem}.jpg"
            src_lbl_f = train_lbl / f"{src_stem}.txt"
            if not src_img_f.exists():
                continue
            shutil.copy2(src_img_f, train_img / f"{dst_stem}.jpg")
            shutil.copy2(src_lbl_f, train_lbl / f"{dst_stem}.txt")
            added += 1
    return added


# ── Class distribution report ─────────────────────────────────────────────────

def report_distribution(train_lbl: Path) -> dict[int, int]:
    class_to_stems: dict[int, list[str]] = defaultdict(list)
    for txt in train_lbl.glob("*.txt"):
        classes_present = {int(line.split()[0]) for line in txt.read_text().splitlines() if line.strip()}
        for cls in classes_present:
            class_to_stems[cls].append(txt.stem)
    counts = {cls: len(stems) for cls, stems in class_to_stems.items()}
    max_count = max(counts.values()) if counts else 1
    print(f"  {'Class':<20} {'Images':>8}  {'ratio':>8}")
    for cls in sorted(counts):
        n = counts[cls]
        print(f"  {CLS_NAMES.get(cls, str(cls)):<20} {n:>8}  {max_count/n:>7.2f}x")
    return counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"CODEBRIM zip not found: {ZIP_PATH}\n"
            "Download from: https://zenodo.org/records/2620293"
        )
    if not CRACK_SRC.exists():
        raise FileNotFoundError(
            f"Crack data not found: {CRACK_SRC}\n"
            "Run: python src/detection/expand_manual_labels.py"
        )

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True)
        (OUT_DIR / split / "labels").mkdir(parents=True)

    train_img = OUT_DIR / "train" / "images"
    train_lbl = OUT_DIR / "train" / "labels"
    val_img   = OUT_DIR / "val"   / "images"
    val_lbl   = OUT_DIR / "val"   / "labels"

    # ── Step 1: CODEBRIM — tile full-resolution images ────────────────────────
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

    paired = sorted(set(xmls) & set(imgs))
    print(f"  Paired (xml+jpg): {len(paired)}")

    random.shuffle(paired)
    n_val   = max(1, int(len(paired) * VAL_FRAC))
    val_set = set(paired[:n_val])

    written_train = written_val = skipped = 0

    print(f"\nTiling CODEBRIM images ({TILE_SIZE}px, stride {TILE_STRIDE}px) ...")
    for stem in paired:
        try:
            xml_data = read_zip_entry(ZIP_PATH, xmls[stem])
            img_w, img_h, abs_boxes = parse_xml(xml_data)
        except Exception:
            skipped += 1
            continue

        if not abs_boxes:
            skipped += 1
            continue

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

        actual_h, actual_w = img.shape[:2]
        if img_w > 0 and img_h > 0 and (actual_w, actual_h) != (img_w, img_h):
            sx = actual_w / img_w
            sy = actual_h / img_h
            abs_boxes = [
                (cls, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                for cls, x1, y1, x2, y2 in abs_boxes
            ]

        tiles = generate_tiles(img, abs_boxes)
        del img

        if not tiles:
            skipped += 1
            continue

        is_val  = stem in val_set
        out_img = val_img   if is_val else train_img
        out_lbl = val_lbl   if is_val else train_lbl

        for idx, (tile, tile_boxes) in enumerate(tiles):
            name = f"codebrim__{stem}__t{idx:03d}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), tile)
            (out_lbl / f"{name}.txt").write_text(fmt(tile_boxes))
            if is_val: written_val   += 1
            else:      written_train += 1

    print(f"  Written — train: {written_train}  val: {written_val}  skipped: {skipped}")

    # ── Step 2: Copy Kaggle crack data ────────────────────────────────────────
    print("\nCopying Kaggle crack data from data/expanded_manual/ ...")
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
    written_train += crack_train

    # ── Step 3: Distribution report ───────────────────────────────────────────
    print("\nClass distribution (train) ...")
    report_distribution(train_lbl)

    total_val = written_val + crack_val
    print(f"\nFinal dataset : {written_train} train  {total_val} val")
    print(f"Output        : {OUT_DIR}")


if __name__ == "__main__":
    main()
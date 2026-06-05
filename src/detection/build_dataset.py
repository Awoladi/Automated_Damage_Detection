"""
BIMInspect — Combined Dataset Builder (Manual + CODEBRIM)

Sources:
  1. Manual labels (Label Studio YOLO exports) — all 5 classes, high quality
     -> 8-way augmentation applied to train images
     -> 80% train, 20% val
  2. CODEBRIM auto-labels (from extracted folder) — all 5 classes
     -> added to train only (auto-labels not suitable for val evaluation)
     -> no extra augmentation (already large and varied)

Balancing: all classes repeated to match the largest class count.
Val set is 100% manually labeled -> reliable mAP estimate.

Output:
  data/expanded_multiclass/
    train/images/ + train/labels/
    val/images/   + val/labels/
  models/configs/dataset_multiclass.yaml
"""

from __future__ import annotations
import random
import shutil
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT         = Path(__file__).resolve().parents[2]
DOWNLOADS    = Path("C:/Users/maxim/Downloads")
TO_LABEL     = ROOT / "data" / "to_label"
CODEBRIM_DIR = ROOT / "data" / "raw" / "CODEBRIM_original_extracted"
ROBOFLOW_DIR = ROOT / "data" / "raw" / "roboflow"
OUT_DIR      = ROOT / "data" / "expanded_multiclass"
YAML_PATH    = ROOT / "models" / "configs" / "dataset_multiclass.yaml"

# Roboflow datasets: map their class IDs to ours (None = skip)
ROBOFLOW_DATASETS = [
    {
        "zip": "Concrete 5November Final 2.v1-concrete-10nov-v1.yolov11.zip",
        "mapping": {0: None, 1: 4, 2: 0, 3: 2, 4: 3, 5: 1},
        # Control Point→skip, Corrosion→4, Crack→0, Efflorescence→2, Exposed Rebar→3, Spalling→1
    },
    {
        "zip": "corrosion.v11i.yolov11.zip",
        "mapping": {0: 4, 1: 4, 2: 4},
        # modest/protection failure/severe → all corrosion (4)
    },
    {
        "zip": "Efflorescence-Det.v1i.yolov11.zip",
        "mapping": {0: 2},  # Efflorescence → efflorescence
    },
    {
        "zip": "Exposure-Det.v1i.yolov11.zip",
        "mapping": {0: 3},  # Exposure-AyCr → exposed_bars
    },
    {
        "zip": "Spalling-Det.v2i.yolov11.zip",
        "mapping": {0: 1},  # Spalling → spallation
    },
]

VAL_FRAC       = 0.20
TILE_SIZE      = 640
TILE_STRIDE    = 480
MIN_VISIBILITY = 0.25
SEED           = 42

CLASSES = {
    "crack":          0,
    "spallation":     1,
    "efflorescence":  2,
    "exposed_bars":   3,
    "corrosion":      4,
}
CLS_NAMES = {v: k for k, v in CLASSES.items()}

CODEBRIM_TAG_TO_CLS = {
    "Crack":          0,
    "Spallation":     1,
    "Efflorescence":  2,
    "ExposedBars":    3,
    "CorrosionStain": 4,
}

random.seed(SEED)
_counter = 0


# ── YOLO label helpers ────────────────────────────────────────────────────────

def parse_labels(text: str) -> list[tuple]:
    rows = []
    for line in text.strip().splitlines():
        p = line.split()
        if len(p) == 5:
            rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows

def fmt_labels(rows: list[tuple]) -> str:
    return "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for c, cx, cy, w, h in rows)


# ── 8-way geometric augmentation ─────────────────────────────────────────────

def aug_original(img, b):   return img.copy(), b
def aug_flip_h(img, b):     return cv2.flip(img, 1),  [(c,1-cx,cy,w,h) for c,cx,cy,w,h in b]
def aug_flip_v(img, b):     return cv2.flip(img, 0),  [(c,cx,1-cy,w,h) for c,cx,cy,w,h in b]
def aug_rot90(img, b):      return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), \
                                   [(c,cy,1-cx,h,w) for c,cx,cy,w,h in b]
def aug_rot180(img, b):     return cv2.rotate(img, cv2.ROTATE_180), \
                                   [(c,1-cx,1-cy,w,h) for c,cx,cy,w,h in b]
def aug_rot270(img, b):     return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), \
                                   [(c,1-cy,cx,h,w) for c,cx,cy,w,h in b]
def aug_transpose(img, b):  return cv2.transpose(img), [(c,cy,cx,h,w) for c,cx,cy,w,h in b]
def aug_transverse(img, b):
    i2, b2 = aug_rot90(img, b); return aug_transpose(i2, b2)

AUGMENTS = [aug_original, aug_flip_h, aug_flip_v,
            aug_rot90, aug_rot180, aug_rot270,
            aug_transpose, aug_transverse]


# ── Label Studio filename decoder ─────────────────────────────────────────────

def decode_label_name(stem: str) -> str:
    without_hash = stem.split("__", 1)[-1]
    decoded = urllib.parse.unquote(without_hash)
    return decoded.replace("\\", "/").split("/")[-1]


# ── CODEBRIM XML parser + tiler ───────────────────────────────────────────────

def parse_xml(xml_path: Path):
    root  = ET.parse(str(xml_path)).getroot()
    img_w = int(root.findtext("size/width",  0))
    img_h = int(root.findtext("size/height", 0))
    boxes = []
    for obj in root.findall("object"):
        defect = obj.find("Defect")
        if defect is None: continue
        xmin = int(obj.findtext("bndbox/xmin", 0))
        ymin = int(obj.findtext("bndbox/ymin", 0))
        xmax = int(obj.findtext("bndbox/xmax", 0))
        ymax = int(obj.findtext("bndbox/ymax", 0))
        if xmax <= xmin or ymax <= ymin: continue
        for tag, cls_id in CODEBRIM_TAG_TO_CLS.items():
            if defect.findtext(tag, "0").strip() == "1":
                boxes.append((cls_id, xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes

def tile_image(img: np.ndarray, abs_boxes: list) -> list[tuple]:
    h, w = img.shape[:2]
    def anchors(total):
        pts = list(range(0, total - TILE_SIZE, TILE_STRIDE))
        pts.append(max(0, total - TILE_SIZE))
        return sorted(set(pts))
    results = []
    for y0 in anchors(h):
        for x0 in anchors(w):
            x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
            tile_boxes = []
            for cls, bx1, by1, bx2, by2 in abs_boxes:
                ix1, ix2 = max(bx1, x0), min(bx2, x1)
                iy1, iy2 = max(by1, y0), min(by2, y1)
                if ix2 <= ix1 or iy2 <= iy1: continue
                orig_area = max(1.0, float((bx2-bx1)*(by2-by1)))
                if (ix2-ix1)*(iy2-iy1)/orig_area < MIN_VISIBILITY: continue
                cx  = ((ix1+ix2)/2 - x0) / TILE_SIZE
                cy  = ((iy1+iy2)/2 - y0) / TILE_SIZE
                bw  = (ix2-ix1) / TILE_SIZE
                bh  = (iy2-iy1) / TILE_SIZE
                tile_boxes.append((cls, cx, cy, bw, bh))
            if tile_boxes:
                results.append((img[y0:y1, x0:x1].copy(), tile_boxes))
    return results


# ── Write helpers ─────────────────────────────────────────────────────────────

def _write(img: np.ndarray, lbl: str, split: str, name: str):
    cv2.imwrite(str(OUT_DIR / split / "images" / f"{name}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT_DIR / split / "labels" / f"{name}.txt").write_text(lbl)

def write_augmented(img_path: Path, lbl: str, split: str):
    global _counter
    img   = cv2.imread(str(img_path))
    if img is None: return
    boxes = parse_labels(lbl)
    for i, fn in enumerate(AUGMENTS):
        aug_img, aug_boxes = fn(img, boxes)
        _counter += 1
        _write(aug_img, fmt_labels(aug_boxes), split, f"m_{img_path.stem}_a{i}_{_counter:06d}")

def write_tile(img: np.ndarray, lbl: str):
    global _counter
    _counter += 1
    _write(img, lbl, "train", f"c_{_counter:06d}")

def write_copy(img_path: Path, lbl: str, split: str):
    global _counter
    img = cv2.imread(str(img_path))
    if img is None: return
    _counter += 1
    _write(img, lbl, split, f"b_{img_path.stem}_{_counter:06d}")


# ── Step 1: Load manual labels ────────────────────────────────────────────────

def load_manual():
    train_by_cls: dict[str, list] = {}
    val_pairs:    list            = []
    print("Step 1: Manual labels\n")
    for cls_name, cls_id in CLASSES.items():
        zip_path  = DOWNLOADS / f"{cls_name}.zip"
        img_index = {p.stem: p for p in (TO_LABEL / cls_name).glob("*.jpg")} \
                    if (TO_LABEL / cls_name).exists() else {}
        if not zip_path.exists():
            print(f"  [{cls_name}] zip not found -- skipping"); continue
        pairs = []
        with zipfile.ZipFile(zip_path) as zf:
            for entry in zf.namelist():
                if not (entry.startswith("labels/") and entry.endswith(".txt")): continue
                img_stem = decode_label_name(Path(entry).stem)
                img_path = img_index.get(img_stem)
                if img_path is None: continue
                rows = [(cls_id, cx, cy, w, h)
                        for _, cx, cy, w, h in parse_labels(zf.read(entry).decode())]
                if rows:
                    pairs.append((img_path, fmt_labels(rows)))
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * VAL_FRAC))
        val_pairs.extend(pairs[:n_val])
        train_by_cls[cls_name] = pairs[n_val:]
        print(f"  [{cls_name}]  {len(pairs[n_val:])} train  {n_val} val")
    return train_by_cls, val_pairs


# ── Step 2: Load Roboflow datasets ───────────────────────────────────────────

def load_roboflow() -> list[tuple[np.ndarray, str]]:
    """Reads all Roboflow zips, remaps class IDs, returns (img, label) pairs."""
    if not ROBOFLOW_DIR.exists():
        print("  Roboflow dir not found -- skipping"); return []

    print("\nStep 2: Loading Roboflow datasets ...")
    results: list[tuple[np.ndarray, str]] = []

    for ds in ROBOFLOW_DATASETS:
        zip_path = ROBOFLOW_DIR / ds["zip"]
        if not zip_path.exists():
            print(f"  NOT FOUND: {ds['zip']}"); continue
        mapping: dict[int, int | None] = ds["mapping"]
        pairs: list[tuple[np.ndarray, str]] = []

        with zipfile.ZipFile(zip_path) as zf:
            # collect all image entries from train + valid + test splits
            img_entries = [e for e in zf.namelist()
                           if e.lower().endswith((".jpg", ".jpeg", ".png"))
                           and not e.startswith("__MACOSX")]
            for img_entry in img_entries:
                lbl_entry = img_entry.replace("/images/", "/labels/")
                lbl_entry = lbl_entry.rsplit(".", 1)[0] + ".txt"
                if lbl_entry not in zf.namelist():
                    continue
                raw_lbl = zf.read(lbl_entry).decode("utf-8")
                remapped = []
                for line in raw_lbl.strip().splitlines():
                    p = line.split()
                    if len(p) < 5: continue
                    src_cls = int(p[0])
                    dst_cls = mapping.get(src_cls)
                    if dst_cls is None: continue  # skip unwanted classes
                    remapped.append(f"{dst_cls} {' '.join(p[1:])}")
                if not remapped: continue
                img_bytes = zf.read(img_entry)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue
                # Resize to 640×640 if needed
                h, w = img.shape[:2]
                if w != 640 or h != 640:
                    img = cv2.resize(img, (640, 640))
                pairs.append((img, "\n".join(remapped)))

        cls_counts: dict[int, int] = defaultdict(int)
        for _, lbl in pairs:
            for c in {int(l.split()[0]) for l in lbl.splitlines() if l.strip()}:
                cls_counts[c] += 1
        print(f"  {ds['zip'][:40]:<40}  {len(pairs)} images")
        for c in sorted(cls_counts):
            print(f"    {CLS_NAMES[c]:<20} {cls_counts[c]}")
        results.extend(pairs)

    return results


# ── Step 3: Tile CODEBRIM ─────────────────────────────────────────────────────

def load_codebrim():
    if not CODEBRIM_DIR.exists():
        print("  CODEBRIM_original_extracted not found -- skipping"); return []
    print("\nStep 3: Tiling CODEBRIM (all 5 classes)")
    xmls = sorted(CODEBRIM_DIR.glob("*.xml"))
    tiles_out = []
    skipped   = 0
    cls_counts: dict[int, int] = defaultdict(int)
    for xml_path in xmls:
        jpg_path = xml_path.with_suffix(".jpg")
        if not jpg_path.exists(): skipped += 1; continue
        try:
            img_w, img_h, abs_boxes = parse_xml(xml_path)
        except Exception: skipped += 1; continue
        if not abs_boxes: skipped += 1; continue
        img = cv2.imread(str(jpg_path))
        if img is None: skipped += 1; continue
        actual_h, actual_w = img.shape[:2]
        if img_w > 0 and img_h > 0 and (actual_w, actual_h) != (img_w, img_h):
            sx, sy = actual_w/img_w, actual_h/img_h
            abs_boxes = [(c, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                         for c, x1, y1, x2, y2 in abs_boxes]
        for tile, tile_boxes in tile_image(img, abs_boxes):
            lbl = fmt_labels(tile_boxes)
            tiles_out.append((tile, lbl))
            for c in {b[0] for b in tile_boxes}:
                cls_counts[c] += 1
        del img
    print(f"  {len(tiles_out)} tiles  ({skipped} skipped)")
    for c in sorted(cls_counts):
        print(f"    {CLS_NAMES[c]:<20} {cls_counts[c]}")
    return tiles_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _counter
    _counter = 0

    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True)
        (OUT_DIR / split / "labels").mkdir(parents=True)

    manual_train, manual_val = load_manual()
    roboflow_pairs            = load_roboflow()
    codebrim_tiles            = load_codebrim()

    # Val: manual only, no augmentation
    print(f"\nWriting {len(manual_val)} val images ...")
    for img_path, lbl in manual_val:
        write_copy(img_path, lbl, "val")

    # Train: manual with 8-way augmentation
    print("\nAugmenting manual train images (x8) ...")
    aug_by_cls: dict[str, list] = {}
    for cls_name, pairs in manual_train.items():
        aug_by_cls[cls_name] = pairs
        for img_path, lbl in pairs:
            write_augmented(img_path, lbl, "train")
        print(f"  [{cls_name}]  {len(pairs)} x 8 = {len(pairs)*8}")

    # Train: CODEBRIM tiles
    # Roboflow images
    print(f"\nWriting {len(roboflow_pairs)} Roboflow images ...")
    for rf_img, lbl in roboflow_pairs:
        write_tile(rf_img, lbl)

    # CODEBRIM tiles
    print(f"\nWriting {len(codebrim_tiles)} CODEBRIM tiles ...")
    for tile_img, lbl in codebrim_tiles:
        write_tile(tile_img, lbl)

    # Count per class
    cls_counts: dict[int, int] = defaultdict(int)
    for txt in (OUT_DIR / "train" / "labels").glob("*.txt"):
        for c in {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}:
            cls_counts[c] += 1

    print("\nBefore balancing:")
    for c in sorted(cls_counts):
        print(f"  {CLS_NAMES[c]:<20} {cls_counts[c]:>6}")

    # Balance: repeat minority classes using manual images
    print("\nBalancing ...")
    target = max(cls_counts.values())
    for cls_name, pairs in aug_by_cls.items():
        cls_id  = CLASSES[cls_name]
        current = cls_counts[cls_id]
        needed  = target - current
        if needed <= 0:
            print(f"  [{cls_name}]  {current} -- at target"); continue
        pool = pairs.copy(); random.shuffle(pool)
        for i in range(needed):
            write_copy(pool[i % len(pool)][0], pool[i % len(pool)][1], "train")
        print(f"  [{cls_name}]  {current} -> {current+needed}  (+{needed})")

    # Final report
    final: dict[int, int] = defaultdict(int)
    for txt in (OUT_DIR / "train" / "labels").glob("*.txt"):
        for c in {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}:
            final[c] += 1

    train_n = len(list((OUT_DIR / "train" / "images").glob("*.jpg")))
    val_n   = len(list((OUT_DIR / "val"   / "images").glob("*.jpg")))
    print(f"\nFinal class distribution (train):")
    for c in sorted(final):
        print(f"  {CLS_NAMES[c]:<20} {final[c]:>6}")
    print(f"\nTotal: {train_n} train  {val_n} val")

    YAML_PATH.write_text(
        f"path: {ROOT.as_posix()}/data/expanded_multiclass\n"
        f"train: train/images\nval:   val/images\n\n"
        f"nc: {len(CLASSES)}\nnames: {list(CLASSES.keys())}\n"
    )
    print(f"Dataset YAML updated: {YAML_PATH}")


if __name__ == "__main__":
    main()

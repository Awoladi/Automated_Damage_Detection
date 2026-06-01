"""
BIMInspect — Manual Label Dataset Builder

Unpacks Label Studio YOLO exports (one zip per class), matches each label
file to its source image in data/to_label/, remaps class IDs to the global
scheme, applies 8-way geometric augmentation to every training image, then
balances all classes to the same count by repeating images from the
underrepresented classes.

Input:
  C:/Users/maxim/Downloads/{crack,spallation,efflorescence,exposed_bars,corrosion}.zip
  data/to_label/<class>/  (source images)

Output:
  data/expanded_multiclass/
    train/images/ + train/labels/
    val/images/   + val/labels/
  models/configs/dataset_multiclass.yaml  (updated)
"""

from __future__ import annotations
import random
import shutil
import urllib.parse
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT      = Path(__file__).resolve().parents[2]
DOWNLOADS = Path("C:/Users/maxim/Downloads")
TO_LABEL  = ROOT / "data" / "to_label"
OUT_DIR   = ROOT / "data" / "expanded_multiclass"
YAML_PATH = ROOT / "models" / "configs" / "dataset_multiclass.yaml"

VAL_FRAC = 0.15
SEED     = 42

CLASSES = {
    "crack":          0,
    "spallation":     1,
    "efflorescence":  2,
    "exposed_bars":   3,
    "corrosion":      4,
}

random.seed(SEED)


# ── YOLO bbox helpers ─────────────────────────────────────────────────────────

def parse_labels(text: str) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in text.strip().splitlines():
        p = line.split()
        if len(p) == 5:
            rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows

def fmt_labels(rows: list[tuple]) -> str:
    return "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows)


# ── 8-way geometric augmentation (closed-form bbox math) ─────────────────────

def aug_original(img, boxes):
    return img.copy(), boxes

def aug_flip_h(img, boxes):
    return cv2.flip(img, 1), [(c, 1-cx, cy, w, h) for c, cx, cy, w, h in boxes]

def aug_flip_v(img, boxes):
    return cv2.flip(img, 0), [(c, cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]

def aug_rot90(img, boxes):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), \
           [(c, cy, 1-cx, h, w) for c, cx, cy, w, h in boxes]

def aug_rot180(img, boxes):
    return cv2.rotate(img, cv2.ROTATE_180), \
           [(c, 1-cx, 1-cy, w, h) for c, cx, cy, w, h in boxes]

def aug_rot270(img, boxes):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), \
           [(c, 1-cy, cx, h, w) for c, cx, cy, w, h in boxes]

def aug_transpose(img, boxes):
    return cv2.transpose(img), [(c, cy, cx, h, w) for c, cx, cy, w, h in boxes]

def aug_transverse(img, boxes):
    img_t, boxes_t = aug_rot90(img, boxes)
    return aug_transpose(img_t, boxes_t)

AUGMENTS = [aug_original, aug_flip_h, aug_flip_v,
            aug_rot90, aug_rot180, aug_rot270,
            aug_transpose, aug_transverse]


# ── Label Studio filename decoder ─────────────────────────────────────────────

def decode_label_name(label_stem: str) -> str:
    without_hash = label_stem.split("__", 1)[-1]
    decoded = urllib.parse.unquote(without_hash)
    return decoded.replace("\\", "/").split("/")[-1]


# ── Per-class extraction ──────────────────────────────────────────────────────

def load_class(cls_name: str, cls_id: int, zip_path: Path) -> tuple[list, list]:
    """Returns (train_pairs, val_pairs) where each pair is (img_path, label_text)."""
    img_dir   = TO_LABEL / cls_name
    img_index = {p.stem: p for p in img_dir.glob("*.jpg")} if img_dir.exists() else {}

    pairs: list[tuple[Path, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if not (entry.startswith("labels/") and entry.endswith(".txt")):
                continue
            img_stem = decode_label_name(Path(entry).stem)
            img_path = img_index.get(img_stem)
            if img_path is None:
                continue
            raw = zf.read(entry).decode("utf-8")
            rows = parse_labels(raw)
            if not rows:
                continue
            # Remap class IDs (Label Studio uses 0 per project)
            rows = [(cls_id, cx, cy, w, h) for _, cx, cy, w, h in rows]
            pairs.append((img_path, fmt_labels(rows)))

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    return pairs[n_val:], pairs[:n_val]   # train, val


# ── Write helpers ─────────────────────────────────────────────────────────────

_counter = 0

def write_pair(img_path: Path, label_text: str, split: str,
               suffix: str = "") -> None:
    global _counter
    _counter += 1
    name = f"{img_path.stem}{suffix}_{_counter:06d}"
    img  = cv2.imread(str(img_path))
    if img is None:
        return
    boxes = parse_labels(label_text)
    dst_img = OUT_DIR / split / "images" / f"{name}.jpg"
    dst_lbl = OUT_DIR / split / "labels" / f"{name}.txt"
    cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    dst_lbl.write_text(label_text)

def write_augmented(img_path: Path, label_text: str, split: str) -> None:
    global _counter
    img   = cv2.imread(str(img_path))
    if img is None:
        return
    boxes = parse_labels(label_text)
    for i, fn in enumerate(AUGMENTS):
        aug_img, aug_boxes = fn(img, boxes)
        _counter += 1
        name    = f"{img_path.stem}_a{i}_{_counter:06d}"
        dst_img = OUT_DIR / split / "images" / f"{name}.jpg"
        dst_lbl = OUT_DIR / split / "labels" / f"{name}.txt"
        cv2.imwrite(str(dst_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        dst_lbl.write_text(fmt_labels(aug_boxes))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _counter
    _counter = 0

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True)
        (OUT_DIR / split / "labels").mkdir(parents=True)

    print("Loading labeled data from Label Studio exports ...\n")

    train_by_cls: dict[str, list[tuple[Path, str]]] = {}
    val_pairs:    list[tuple[Path, str]] = []

    for cls_name, cls_id in CLASSES.items():
        zip_path = DOWNLOADS / f"{cls_name}.zip"
        if not zip_path.exists():
            print(f"  [{cls_name}] zip not found — skipping")
            continue
        train_pairs, val = load_class(cls_name, cls_id, zip_path)
        train_by_cls[cls_name] = train_pairs
        val_pairs.extend(val)
        print(f"  [{cls_name}]  {len(train_pairs)} train  {len(val)} val")

    # ── Val: write originals only (no augmentation) ───────────────────────────
    print(f"\nWriting {len(val_pairs)} val images ...")
    for img_path, label_text in val_pairs:
        write_pair(Path(img_path), label_text, "val")

    # ── Train: 8-way augmentation then balance ────────────────────────────────
    print("\nApplying 8-way augmentation to train set ...")
    aug_by_cls: dict[str, list[tuple[Path, str]]] = {}
    for cls_name, pairs in train_by_cls.items():
        aug_by_cls[cls_name] = pairs   # keep originals for repeat-sampling
        for img_path, label_text in pairs:
            write_augmented(Path(img_path), label_text, "train")
        aug_count = len(pairs) * len(AUGMENTS)
        print(f"  [{cls_name}]  {len(pairs)} × {len(AUGMENTS)} = {aug_count} images")

    # ── Balance: count images per class, repeat minority ─────────────────────
    print("\nBalancing classes ...")
    cls_counts: dict[int, int] = defaultdict(int)
    cls_names_inv = {v: k for k, v in CLASSES.items()}
    for txt in (OUT_DIR / "train" / "labels").glob("*.txt"):
        seen = {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}
        for c in seen:
            cls_counts[c] += 1

    target = max(cls_counts.values())
    for cls_name, pairs in aug_by_cls.items():
        cls_id  = CLASSES[cls_name]
        current = cls_counts[cls_id]
        needed  = target - current
        if needed <= 0:
            print(f"  [{cls_name}]  {current} — at target")
            continue
        pool = pairs.copy()
        random.shuffle(pool)
        added = 0
        for i in range(needed):
            img_path, label_text = pool[i % len(pool)]
            write_pair(Path(img_path), label_text, "train", suffix="_bal")
            added += 1
        print(f"  [{cls_name}]  {current} -> {current + added}  (+{added} repeated)")

    # ── Final report ──────────────────────────────────────────────────────────
    final_counts: dict[int, int] = defaultdict(int)
    for txt in (OUT_DIR / "train" / "labels").glob("*.txt"):
        seen = {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}
        for c in seen:
            final_counts[c] += 1

    train_total = len(list((OUT_DIR / "train" / "images").glob("*.jpg")))
    val_total   = len(list((OUT_DIR / "val"   / "images").glob("*.jpg")))

    print(f"\nFinal class distribution (train):")
    for c in sorted(final_counts):
        print(f"  {cls_names_inv[c]:<20} {final_counts[c]:>6}")
    print(f"\nTotal: {train_total} train  {val_total} val")

    yaml_content = f"""path: {ROOT.as_posix()}/data/expanded_multiclass
train: train/images
val:   val/images

nc: {len(CLASSES)}
names: {list(CLASSES.keys())}
"""
    YAML_PATH.write_text(yaml_content)
    print(f"Dataset YAML updated: {YAML_PATH}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

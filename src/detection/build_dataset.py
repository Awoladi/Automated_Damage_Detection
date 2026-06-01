"""
BIMInspect — Manual Label Dataset Builder

Unpacks Label Studio YOLO exports (one zip per class), matches each label
file to its source image in data/to_label/, remaps class IDs to the global
scheme, splits 85/15 train/val, and writes data/expanded_multiclass/.

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
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DOWNLOADS   = Path("C:/Users/maxim/Downloads")
TO_LABEL    = ROOT / "data" / "to_label"
OUT_DIR     = ROOT / "data" / "expanded_multiclass"
YAML_PATH   = ROOT / "models" / "configs" / "dataset_multiclass.yaml"

VAL_FRAC = 0.15
SEED     = 42

# Global class mapping — order matters for YOLO
CLASSES = {
    "crack":          0,
    "spallation":     1,
    "efflorescence":  2,
    "exposed_bars":   3,
    "corrosion":      4,
}

random.seed(SEED)


def decode_label_name(label_stem: str) -> str:
    """
    Label Studio encodes the source path in the label filename:
      <hash>__<url-encoded-path-without-extension>
    Returns just the image filename stem (last path component after decoding).
    """
    # Remove the leading hash prefix (everything up to and including first __)
    without_hash = label_stem.split("__", 1)[-1]
    decoded = urllib.parse.unquote(without_hash)   # %5C -> \, %2F -> /
    # Take the last component as the image stem
    img_stem = decoded.replace("\\", "/").split("/")[-1]
    return img_stem


def build_class(cls_name: str, cls_id: int, zip_path: Path,
                all_items: list[tuple[str, str, str]]) -> int:
    """
    Extract labels from zip, find matching image in to_label/<cls_name>/,
    append (img_path, label_text, split) tuples to all_items.
    Returns count of matched pairs.
    """
    img_dir = TO_LABEL / cls_name
    if not img_dir.exists():
        print(f"  WARNING: {img_dir} not found — skipping {cls_name}")
        return 0

    # Index available images by stem
    img_index = {p.stem: p for p in img_dir.glob("*.jpg")}

    matched = 0
    with zipfile.ZipFile(zip_path) as zf:
        label_entries = [e for e in zf.namelist()
                         if e.startswith("labels/") and e.endswith(".txt")]

        pairs: list[tuple[Path, str]] = []
        for entry in label_entries:
            label_stem = Path(entry).stem
            img_stem   = decode_label_name(label_stem)

            img_path = img_index.get(img_stem)
            if img_path is None:
                continue

            raw_label = zf.read(entry).decode("utf-8")
            # Remap class IDs: Label Studio uses 0 for every class within its
            # per-project single-class setup → replace with global cls_id
            remapped_lines = []
            for line in raw_label.strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                parts[0] = str(cls_id)
                remapped_lines.append(" ".join(parts))

            if not remapped_lines:
                continue

            pairs.append((img_path, "\n".join(remapped_lines)))
            matched += 1

    # Shuffle and split
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    for i, (img_path, label_text) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        all_items.append((str(img_path), label_text, split))

    return matched


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True)
        (OUT_DIR / split / "labels").mkdir(parents=True)

    all_items: list[tuple[str, str, str]] = []

    print("Building dataset from Label Studio exports ...\n")
    for cls_name, cls_id in CLASSES.items():
        zip_path = DOWNLOADS / f"{cls_name}.zip"
        if not zip_path.exists():
            print(f"  [{cls_name}] zip not found at {zip_path} — skipping")
            continue
        n = build_class(cls_name, cls_id, zip_path, all_items)
        print(f"  [{cls_name}]  {n} labeled images")

    print(f"\nTotal: {len(all_items)} labeled images")

    # Write to disk
    train_count = val_count = 0
    for i, (img_path, label_text, split) in enumerate(all_items):
        src = Path(img_path)
        name = f"{src.stem}_{i:05d}"
        dst_img = OUT_DIR / split / "images" / f"{name}.jpg"
        dst_lbl = OUT_DIR / split / "labels" / f"{name}.txt"
        shutil.copy2(src, dst_img)
        dst_lbl.write_text(label_text)
        if split == "train":
            train_count += 1
        else:
            val_count += 1

    print(f"Written: {train_count} train  {val_count} val")

    # Class distribution report
    print("\nClass distribution (train):")
    from collections import defaultdict
    cls_counts: dict[int, int] = defaultdict(int)
    for txt in (OUT_DIR / "train" / "labels").glob("*.txt"):
        seen = {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}
        for c in seen:
            cls_counts[c] += 1
    cls_names = {v: k for k, v in CLASSES.items()}
    for c in sorted(cls_counts):
        print(f"  {cls_names[c]:<20} {cls_counts[c]:>5}")

    # Update dataset YAML
    yaml_content = f"""path: {ROOT.as_posix()}/data/expanded_multiclass
train: train/images
val:   val/images

nc: {len(CLASSES)}
names: {list(CLASSES.keys())}
"""
    YAML_PATH.write_text(yaml_content)
    print(f"\nDataset YAML updated: {YAML_PATH}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

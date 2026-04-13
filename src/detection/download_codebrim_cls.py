"""
Downloads CODEBRIM_classification_dataset.zip and extracts 500 images per
damage class with YOLO labels where bbox = full image (the crop IS the defect).

YOLO label format per image: class_id 0.5 0.5 1.0 1.0

Output: data/annotated_multiclass/{class_name}/images/ + labels/
        (replaces the sparse original-image annotations)
"""

import random
import struct
import zipfile
import zlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parents[2]
ZIP_PATH   = ROOT / "data" / "raw" / "CODEBRIM_classification_dataset.zip"
OUT_DIR    = ROOT / "data" / "annotated_multiclass"
ZENODO_URL = (
    "https://zenodo.org/api/records/2620293/files/"
    "CODEBRIM_classification_dataset.zip/content"
)
SAMPLES_PER_CLASS = 500
SEED = 42
ZIP_OFFSET_SHIFT  = 4_294_967_296

# XML tag → (YOLO class id, folder name)
CLASSES = {
    "Spallation":     (1, "spallation"),
    "Efflorescence":  (2, "efflorescence"),
    "ExposedBars":    (3, "exposed_bars"),
    "CorrosionStain": (4, "corrosion"),
    # skip Crack — we already have better crack data
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest.name}")
        return
    print(f"Downloading {dest.name} (~8 GB) ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    print("Download complete.")


def read_entry(zip_path: Path, info: zipfile.ZipInfo) -> bytes:
    """Read a zip entry, auto-detecting the 4-GB offset shift."""
    with open(zip_path, "rb") as f:
        offset = info.header_offset
        f.seek(offset)
        if f.read(4) != b"PK\x03\x04":
            offset = info.header_offset - ZIP_OFFSET_SHIFT
            f.seek(offset)
            if f.read(4) != b"PK\x03\x04":
                raise ValueError(f"PK magic not found for {info.filename}")
        f.seek(offset + 4)
        f.seek(22, 1)
        fname_len = struct.unpack("<H", f.read(2))[0]
        extra_len = struct.unpack("<H", f.read(2))[0]
        f.seek(fname_len + extra_len, 1)
        raw = f.read(info.compress_size)

    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    raise ValueError(f"Unsupported compression: {info.compress_type}")


def main() -> None:
    random.seed(SEED)
    download(ZENODO_URL, ZIP_PATH)

    print("Reading zip index ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        all_entries = {Path(e.filename).name: e for e in zf.infolist()
                       if not e.filename.startswith("__MACOSX")}

        # Find the metadata XML
        xml_entry = next(
            (e for e in zf.infolist()
             if "defects.xml" in e.filename and "__MACOSX" not in e.filename),
            None
        )
        if xml_entry is None:
            raise FileNotFoundError("defects.xml not found in zip")

        print("Parsing XML metadata ...")
        xml_data = read_entry(ZIP_PATH, xml_entry)
        root_el  = ET.fromstring(xml_data)

        # Group image filenames by class (single-class only)
        class_images: dict[str, list[str]] = defaultdict(list)
        for defect in root_el.findall("Defect"):
            name   = defect.get("name", "")
            active = [tag for tag in CLASSES
                      if defect.findtext(tag, "0").strip() == "1"]
            if len(active) == 1:
                class_images[active[0]].append(name)

        # Also build a lookup of all image entries by filename
        img_entries = {
            Path(e.filename).name: e
            for e in zf.infolist()
            if e.filename.endswith(".png") and "__MACOSX" not in e.filename
        }
        print(f"  Total images in zip: {len(img_entries)}")
        for tag, names in class_images.items():
            cls_id, cls_name = CLASSES[tag]
            avail = [n for n in names if n in img_entries]
            print(f"  {cls_name:20s}: {len(avail)} single-class images")

        print("\nExtracting 500 per class ...")
        for tag, names in class_images.items():
            cls_id, cls_name = CLASSES[tag]
            avail  = [n for n in names if n in img_entries]
            sample = random.sample(avail, min(SAMPLES_PER_CLASS, len(avail)))

            img_out = OUT_DIR / cls_name / "images"
            lbl_out = OUT_DIR / cls_name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            # Clear old sparse annotations
            for f in img_out.glob("*.jpg"): f.unlink()
            for f in lbl_out.glob("*.txt"): f.unlink()

            print(f"  {cls_name:20s} — {len(sample)} images")
            for fname in tqdm(sample, leave=False):
                info = img_entries[fname]
                try:
                    data = read_entry(ZIP_PATH, info)
                except Exception:
                    continue
                stem = Path(fname).stem
                (img_out / fname).write_bytes(data)
                # Full-image bbox: class cx cy w h
                (lbl_out / f"{stem}.txt").write_text(
                    f"{cls_id} 0.500000 0.500000 1.000000 1.000000\n"
                )
            print(f"  {cls_name:20s} — done")

    print("\n-- Summary --")
    for _, (cls_id, cls_name) in CLASSES.items():
        imgs = len(list((OUT_DIR / cls_name / "images").glob("*.png")))
        lbls = len(list((OUT_DIR / cls_name / "labels").glob("*.txt")))
        print(f"  {cls_name:20s}: {imgs} images, {lbls} labels")


if __name__ == "__main__":
    main()

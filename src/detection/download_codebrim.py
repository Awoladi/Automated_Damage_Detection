"""
Downloads CODEBRIM classification dataset from Zenodo and extracts
500 images per damage class into data/raw/{class_name}/.

Uses the XML metadata to assign images to classes (pure/dominant label).

Classes extracted:
  - spallation    → data/raw/spallation/
  - corrosion     → data/raw/corrosion/
  - efflorescence → data/raw/efflorescence/
  - exposed_bars  → data/raw/exposed_bars/
  - crack         → data/raw/crack_codebrim/
"""

import random
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
RAW_DIR    = ROOT / "data" / "raw"
ZIP_PATH   = RAW_DIR / "CODEBRIM_classification_dataset.zip"
ZENODO_URL = (
    "https://zenodo.org/api/records/2620293/files/"
    "CODEBRIM_classification_dataset.zip/content"
)
SAMPLES_PER_CLASS = 500
SEED              = 42

# XML tag → output folder name
CLASS_MAP = {
    "Spallation":    "spallation",
    "CorrosionStain":"corrosion",
    "Efflorescence": "efflorescence",
    "ExposedBars":   "exposed_bars",
    "Crack":         "crack_codebrim",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest.name}")
        return
    print(f"Downloading {dest.name} (~8 GB) …")
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


def parse_xml(zf: zipfile.ZipFile, xml_path: str) -> dict[str, list[str]]:
    """Parse defects.xml → {class_tag: [image_filename, ...]}"""
    raw = zf.read(xml_path).decode()
    root = ET.fromstring(raw)
    class_images: dict[str, list[str]] = {k: [] for k in CLASS_MAP}

    for defect in root.findall("Defect"):
        name = defect.get("name", "")
        # Assign image to ALL classes it belongs to (multi-label)
        for tag in CLASS_MAP:
            el = defect.find(tag)
            if el is not None and el.text and el.text.strip() == "1":
                class_images[tag].append(name)

    return class_images


EXTRACTED_DIR = RAW_DIR / "CODEBRIM_extracted"
XML_PATH      = EXTRACTED_DIR / "classification_dataset" / "metadata" / "defects.xml"


def extract_samples(zip_path: Path) -> None:
    random.seed(SEED)

    # ── Step 1: extract raw files if not done ─────────────────────────────────
    defects_dir = EXTRACTED_DIR / "classification_dataset" / "train" / "defects"
    if not defects_dir.exists():
        print("\nExtracting images from zip (one-time, may take a minute) …")
        import subprocess
        subprocess.run([
            "unzip", "-o", str(zip_path),
            "classification_dataset/metadata/defects.xml",
            "classification_dataset/train/defects/*",
            "classification_dataset/test/defects/*",
            "classification_dataset/val/defects/*",
            "-d", str(EXTRACTED_DIR),
        ], check=True)

    # ── Step 2: build filename → path lookup ──────────────────────────────────
    img_lookup: dict[str, Path] = {}
    for img in EXTRACTED_DIR.rglob("*.png"):
        img_lookup[img.name] = img

    # ── Step 3: parse XML ─────────────────────────────────────────────────────
    print("\nParsing XML metadata …")
    tree = ET.parse(XML_PATH)
    root_el = tree.getroot()
    class_images: dict[str, list[str]] = {k: [] for k in CLASS_MAP}

    for defect in root_el.findall("Defect"):
        name = defect.get("name", "")
        for tag in CLASS_MAP:
            el = defect.find(tag)
            if el is not None and el.text and el.text.strip() == "1":
                class_images[tag].append(name)

    print(f"  Total images available: {len(img_lookup)}")
    for tag, filenames in class_images.items():
        print(f"  {CLASS_MAP[tag]:20s}: {len(filenames)} labeled")

    # ── Step 4: copy samples ──────────────────────────────────────────────────
    print("\nCopying samples …")
    for tag, filenames in class_images.items():
        out_dir = RAW_DIR / CLASS_MAP[tag]
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(out_dir.glob("*.png")))
        if existing >= SAMPLES_PER_CLASS:
            print(f"  {CLASS_MAP[tag]:20s} — already has {existing} images, skipping")
            continue

        available = [f for f in filenames if f in img_lookup]
        sample = random.sample(available, min(SAMPLES_PER_CLASS, len(available)))
        print(f"  {CLASS_MAP[tag]:20s} — copying {len(sample)} images")

        for fname in tqdm(sample, desc=f"  {CLASS_MAP[tag]}", leave=False):
            import shutil
            shutil.copy2(img_lookup[fname], out_dir / fname)

        print(f"  {CLASS_MAP[tag]:20s} — done")


def main() -> None:
    download(ZENODO_URL, ZIP_PATH)
    extract_samples(ZIP_PATH)

    print("\n── Class summary ──")
    for out_name in CLASS_MAP.values():
        folder = RAW_DIR / out_name
        count  = len(list(folder.glob("*.png"))) if folder.exists() else 0
        print(f"  {out_name:20s}: {count} images")

    print("\nDone. Delete the zip to free ~8 GB:")
    print(f"  del \"{ZIP_PATH}\"")


if __name__ == "__main__":
    main()

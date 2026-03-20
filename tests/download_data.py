"""
Download the Concrete Crack Images for Classification dataset from Kaggle.

Dataset : arunrk7/surface-crack-detection
  - 40,000 images (227x227 px, RGB)
  - Two classes: Positive (crack) / Negative (no crack)
  - Delivered as a ZIP which is automatically extracted

Usage
-----
    # Activate venv first, then:
    python tests/download_data.py

Kaggle credentials
------------------
Place your kaggle.json token at one of:
  - %USERPROFILE%\\.kaggle\\kaggle.json   (Windows default)
  - KAGGLE_USERNAME + KAGGLE_KEY env vars

To get kaggle.json:
  1. Log in at kaggle.com
  2. Account → Settings → API → "Create New Token"
"""

import os
import sys
import zipfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATASET = "arunrk7/surface-crack-detection"
ZIP_NAME = "surface-crack-detection.zip"


def check_credentials() -> None:
    """Abort early with a helpful message if no Kaggle credentials are found."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")

    if not kaggle_json.exists() and not has_env:
        print("ERROR: Kaggle credentials not found.")
        print()
        print("Option 1 — kaggle.json file:")
        print(f"  Place kaggle.json at: {kaggle_json}")
        print()
        print("Option 2 — environment variables:")
        print("  set KAGGLE_USERNAME=your_username")
        print("  set KAGGLE_KEY=your_api_key")
        print()
        print("Get your token at: https://www.kaggle.com/settings (API section)")
        sys.exit(1)


def download() -> None:
    check_credentials()

    # Import here so the credential check runs first and gives a clean error
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DIR / ZIP_NAME
    if zip_path.exists():
        print(f"ZIP already present at {zip_path}, skipping download.")
    else:
        print(f"Downloading dataset '{DATASET}' ...")
        api.dataset_download_files(DATASET, path=str(RAW_DIR), quiet=False)
        print("Download complete.")

    # ── Extract ───────────────────────────────────────────────────────────────
    # The ZIP extracts Positive/ and Negative/ directly into RAW_DIR
    already_extracted = (RAW_DIR / "Positive").exists() or (RAW_DIR / "Negative").exists()
    if already_extracted:
        print(f"Already extracted at {RAW_DIR}, skipping extraction.")
    else:
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        print("Extraction complete.")

    # ── Summary ───────────────────────────────────────────────────────────────
    positive = list((RAW_DIR / "Positive").glob("*.jpg"))
    negative = list((RAW_DIR / "Negative").glob("*.jpg"))
    total = len(positive) + len(negative)

    print()
    print("Dataset ready:")
    print(f"  Location : {RAW_DIR}")
    print(f"  Positive (crack)     : {len(positive):,} images")
    print(f"  Negative (no crack)  : {len(negative):,} images")
    print(f"  Total                : {total:,} images")


if __name__ == "__main__":
    download()

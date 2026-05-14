"""
BIMInspect — CODEBRIM Tile Extractor for Manual Labeling

Extracts up to 500 representative 640×640 tiles per damage class from the
CODEBRIM zip. Tiles are selected by highest annotation density for that class
so the images are clear, unambiguous examples — easy to label.

Output:
  data/to_label/
    crack/          up to 500 tiles
    spallation/     up to 500 tiles
    efflorescence/  up to 500 tiles
    exposed_bars/   up to 500 tiles
    corrosion/      up to 500 tiles

Import the folders into Label Studio as separate projects (one per class),
draw bounding boxes, export as YOLO format to data/annotated/<class>/
"""

from __future__ import annotations
import struct
import zipfile
import zlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT     = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "data" / "raw" / "CODEBRIM_original_images.zip"
OUT_DIR  = ROOT / "data" / "to_label"

TILE_SIZE      = 640
TILE_STRIDE    = 480
MIN_VISIBILITY = 0.25
TARGET         = 500   # tiles per class
ZIP_SHIFT      = 4_294_967_296

TAG_TO_CLS = {
    "Crack":          "crack",
    "Spallation":     "spallation",
    "Efflorescence":  "efflorescence",
    "ExposedBars":    "exposed_bars",
    "CorrosionStain": "corrosion",
}


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
    raise ValueError(f"Could not read: {info.filename}")


def parse_xml(data: bytes):
    root = ET.fromstring(data)
    img_w = int(root.findtext("size/width",  0))
    img_h = int(root.findtext("size/height", 0))
    boxes = []   # (class_name, xmin, ymin, xmax, ymax)
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
        for tag, cls_name in TAG_TO_CLS.items():
            if defect.findtext(tag, "0").strip() == "1":
                boxes.append((cls_name, xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes


def anchors(total: int) -> list[int]:
    pts = list(range(0, total - TILE_SIZE, TILE_STRIDE))
    pts.append(max(0, total - TILE_SIZE))
    return sorted(set(pts))


def score_tile(x0, y0, boxes, target_cls) -> float:
    """Coverage of target_cls boxes inside this tile (0-1)."""
    x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
    score = 0.0
    for cls_name, bx1, by1, bx2, by2 in boxes:
        if cls_name != target_cls:
            continue
        ix1, ix2 = max(bx1, x0), min(bx2, x1)
        iy1, iy2 = max(by1, y0), min(by2, y1)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        orig_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
        vis = (ix2 - ix1) * (iy2 - iy1) / orig_area
        if vis >= MIN_VISIBILITY:
            score += vis
    return score


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"CODEBRIM zip not found: {ZIP_PATH}")

    if OUT_DIR.exists():
        import shutil
        shutil.rmtree(OUT_DIR)
    for cls_name in TAG_TO_CLS.values():
        (OUT_DIR / cls_name).mkdir(parents=True)

    print("Reading CODEBRIM zip index ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        entries = zf.infolist()
    xmls = {Path(e.filename).stem: e for e in entries
            if e.filename.endswith(".xml") and "__MACOSX" not in e.filename}
    imgs = {Path(e.filename).stem: e for e in entries
            if e.filename.endswith(".jpg") and "__MACOSX" not in e.filename}

    paired = sorted(set(xmls) & set(imgs))
    print(f"  Paired images: {len(paired)}\n")

    # Per class: list of (score, stem, x0, y0)
    candidates: dict[str, list[tuple]] = defaultdict(list)

    print("Scanning tiles for each class ...")
    for stem in paired:
        try:
            xml_data = read_zip_entry(ZIP_PATH, xmls[stem])
            img_w, img_h, boxes = parse_xml(xml_data)
        except Exception:
            continue
        if not boxes:
            continue

        # Infer actual image size if XML size is 0
        if img_w == 0 or img_h == 0:
            try:
                raw = read_zip_entry(ZIP_PATH, imgs[stem])
                arr = np.frombuffer(raw, np.uint8)
                tmp = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if tmp is not None:
                    img_h, img_w = tmp.shape[:2]
            except Exception:
                continue

        classes_present = {cls for cls, *_ in boxes}
        for cls_name in classes_present:
            for y0 in anchors(img_h):
                for x0 in anchors(img_w):
                    sc = score_tile(x0, y0, boxes, cls_name)
                    if sc > 0:
                        candidates[cls_name].append((sc, stem, x0, y0))

    print("Extracting top tiles per class ...")
    for cls_name, cands in sorted(candidates.items()):
        cands.sort(key=lambda t: -t[0])   # highest score first
        selected = cands[:TARGET]
        print(f"  {cls_name:<20} {len(selected):>4} tiles  (pool: {len(cands)})")

        by_stem: dict[str, list[tuple]] = defaultdict(list)
        for sc, stem, x0, y0 in selected:
            by_stem[stem].append((x0, y0))

        saved = 0
        for stem, coords in by_stem.items():
            try:
                raw = read_zip_entry(ZIP_PATH, imgs[stem])
                arr = np.frombuffer(raw, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                for x0, y0 in coords:
                    x1 = min(x0 + TILE_SIZE, w)
                    y1 = min(y0 + TILE_SIZE, h)
                    tile = img[y0:y1, x0:x1]
                    fname = OUT_DIR / cls_name / f"{stem}__x{x0}_y{y0}.jpg"
                    cv2.imwrite(str(fname), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    saved += 1
            except Exception:
                continue

        print(f"    -> saved {saved} images to data/to_label/{cls_name}/")

    print(f"\nDone. Import each subfolder of data/to_label/ into Label Studio.")
    print("After labeling, export as YOLO format to data/annotated/<class>/")


if __name__ == "__main__":
    main()
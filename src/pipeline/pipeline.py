"""
BIMInspect — End-to-End Pipeline
Connects detector.py and ifc_writer.py into a single callable.

Single-image usage:
    result = run(image_path="photo.jpg", ifc_path="building.ifc")

Batch usage:
    results = run_batch(image_dir="data/raw/Positive", ifc_path="building.ifc")

PipelineResult returned per image:
    image_path      : str
    damage_class    : str        "crack" | "no_crack"
    confidence      : float
    bbox            : tuple | None
    ifc_output_path : Path | None   None if skipped (no_crack + skip_no_damage=True)
    skipped         : bool
    error           : str | None    set if detection or IFC write raised an exception
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.detection.detector import DamageDetector, DetectionResult, DEFAULT_WEIGHTS
from src.bim.ifc_writer import IFCWriter, from_detection_result

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "ifc" / "templates"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Per-image result ───────────────────────────────────────────────────────────

NO_DAMAGE_CLASSES = {"no_crack", "no_damage", ""}

@dataclass
class PipelineResult:
    image_path:      str
    damage_class:    str   = ""
    confidence:      float = 0.0
    bbox:            Optional[tuple[int, int, int, int]] = None
    bbox_normalized: Optional[tuple[float, float, float, float]] = None
    ifc_output_path: Optional[Path] = None
    skipped:         bool  = False
    error:           Optional[str] = None
    all_detections:  list  = None  # [(x1,y1,x2,y2,conf,cls_name), ...]

    def __post_init__(self):
        if self.all_detections is None:
            self.all_detections = []

    def __str__(self) -> str:
        status = "ERROR" if self.error else ("SKIP" if self.skipped else "OK")
        line = f"[{status}] {Path(self.image_path).name}"
        if self.damage_class:
            line += f"  class={self.damage_class}  conf={self.confidence:.4f}"
        if self.bbox:
            line += f"  bbox={self.bbox}"
        if self.ifc_output_path:
            line += f"  -> {self.ifc_output_path.name}"
        if self.error:
            line += f"  ({self.error})"
        return line


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run(
    image_path:      str | Path,
    ifc_path:        Optional[str | Path] = None,
    *,
    weights:         str | Path = DEFAULT_WEIGHTS,
    detector:        Optional[DamageDetector] = None,
    writer:          Optional[IFCWriter] = None,
    ifc_output_name: Optional[str] = None,
    storey_name:     str = "Ground Floor",
    skip_no_damage:  bool = True,
    confidence_threshold: float = 0.5,
) -> PipelineResult:
    """
    Run the full BIMInspect pipeline on a single image.

    Args:
        image_path:           Path to the input image.
        ifc_path:             Path to the IFC template to annotate.
                              - Absolute path: used directly.
                              - Relative path: resolved against ifc/templates/.
                              - None: a minimal IFC file is created from scratch.
        weights:              Path to YOLOv8 weights (default: models/weights/best.pt).
        detector:             Pre-loaded DamageDetector (avoids reloading the model
                              on every call when processing batches).
        writer:               Pre-created IFCWriter (allows accumulating multiple
                              detections into one IFC file across calls).
        ifc_output_name:      Filename for the saved IFC. Auto-generated if None.
        storey_name:          Target IfcBuildingStorey name inside the IFC hierarchy.
        skip_no_damage:       If True, images classified as "no_crack" are not written
                              to the IFC file (default: True).
        confidence_threshold: Detections below this confidence are treated as
                              no-damage regardless of the predicted class.

    Returns:
        PipelineResult with detection output and IFC output path.
    """
    image_path = Path(image_path)
    result = PipelineResult(image_path=str(image_path))

    # ── 1. Detect ──────────────────────────────────────────────────────────────
    try:
        _detector = detector or DamageDetector(weights=weights)
        detection: DetectionResult = _detector.detect(image_path)
    except Exception as exc:
        result.error = f"Detection failed: {exc}"
        traceback.print_exc()
        return result

    result.damage_class    = detection.damage_class
    result.confidence      = detection.confidence
    result.bbox            = detection.bbox
    result.bbox_normalized = detection.bbox_normalized
    result.all_detections  = detection.all_detections

    # ── 2. Threshold + skip check ──────────────────────────────────────────────
    is_damage = (
        detection.damage_class not in NO_DAMAGE_CLASSES
        and detection.confidence >= confidence_threshold
    )

    if not is_damage and skip_no_damage:
        result.skipped = True
        return result

    # ── 3. Write to IFC ────────────────────────────────────────────────────────
    try:
        _writer = writer or _resolve_writer(ifc_path)
        record  = from_detection_result(detection)
        record.storey_name = storey_name
        _writer.write(record)

        # Only save when no shared writer is passed in — callers using a shared
        # writer are responsible for calling writer.save() themselves.
        if writer is None:
            result.ifc_output_path = _writer.save(ifc_output_name)

    except Exception as exc:
        result.error = f"IFC write failed: {exc}"
        traceback.print_exc()

    return result


def run_batch(
    image_dir:       str | Path,
    ifc_path:        Optional[str | Path] = None,
    *,
    weights:         str | Path = DEFAULT_WEIGHTS,
    ifc_output_name: Optional[str] = None,
    storey_name:     str = "Ground Floor",
    skip_no_damage:  bool = True,
    confidence_threshold: float = 0.5,
    recursive:       bool = False,
) -> list[PipelineResult]:
    """
    Run the pipeline on every image in a directory, accumulating all damage
    annotations into a single IFC file.

    Args:
        image_dir:   Directory containing images (scanned for common extensions).
        ifc_path:    IFC template path (see run() docstring).
        recursive:   If True, also scan sub-directories.
        (others)     Same as run().

    Returns:
        List of PipelineResult, one per image.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    glob = "**/*" if recursive else "*"
    images = sorted(
        p for p in image_dir.glob(glob)
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No images found in {image_dir}")
        return []

    print(f"BIMInspect batch: {len(images)} image(s) in {image_dir}")
    print(f"  skip_no_damage={skip_no_damage}  threshold={confidence_threshold}\n")

    # Load model and IFC writer once for all images
    detector = DamageDetector(weights=weights)
    writer   = _resolve_writer(ifc_path)

    results: list[PipelineResult] = []
    damage_count = 0

    for i, img in enumerate(images, 1):
        print(f"[{i:>4}/{len(images)}] {img.name}", end="  ")
        r = run(
            image_path           = img,
            ifc_path             = ifc_path,
            detector             = detector,
            writer               = writer,         # shared — accumulates annotations
            storey_name          = storey_name,
            skip_no_damage       = skip_no_damage,
            confidence_threshold = confidence_threshold,
        )
        results.append(r)
        print(r)
        if not r.skipped and not r.error:
            damage_count += 1

    # Save the shared writer once after all images
    out_path = writer.save(ifc_output_name)
    for r in results:
        if not r.skipped and not r.error:
            r.ifc_output_path = out_path

    _print_summary(results, out_path)
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_writer(ifc_path: Optional[str | Path]) -> IFCWriter:
    """Create an IFCWriter from an optional IFC path."""
    if ifc_path is None:
        return IFCWriter()

    ifc_path = Path(ifc_path)
    if not ifc_path.is_absolute():
        ifc_path = TEMPLATES_DIR / ifc_path

    return IFCWriter(template=ifc_path)


def _print_summary(results: list[PipelineResult], ifc_out: Path) -> None:
    total   = len(results)
    damaged = sum(1 for r in results if not r.skipped and not r.error)
    skipped = sum(1 for r in results if r.skipped)
    errors  = sum(1 for r in results if r.error)

    print("\n" + "=" * 60)
    print(f"  BIMInspect — Batch Complete")
    print(f"  Total images : {total}")
    print(f"  Damage found : {damaged}")
    print(f"  No damage    : {skipped}")
    print(f"  Errors       : {errors}")
    print(f"  IFC output   : {ifc_out}")
    print("=" * 60)


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="BIMInspect — detect structural damage and write to IFC"
    )
    parser.add_argument("image",   help="Image file or directory")
    parser.add_argument("--ifc",   help="IFC template path (relative to ifc/templates/ or absolute)", default=None)
    parser.add_argument("--out",   help="Output IFC filename", default=None)
    parser.add_argument("--storey",help="Target storey name in IFC", default="Ground Floor")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default 0.5)")
    parser.add_argument("--all",   action="store_true", help="Also write no-crack results to IFC")
    parser.add_argument("--recursive", action="store_true", help="Scan sub-directories (batch mode)")
    args = parser.parse_args()

    path = Path(args.image)
    kwargs = dict(
        ifc_path             = args.ifc,
        ifc_output_name      = args.out,
        storey_name          = args.storey,
        confidence_threshold = args.threshold,
        skip_no_damage       = not args.all,
    )

    if path.is_dir():
        run_batch(path, recursive=args.recursive, **kwargs)
    else:
        result = run(path, **kwargs)
        print(result)
        if result.error:
            sys.exit(1)

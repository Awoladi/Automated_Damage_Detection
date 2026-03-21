"""
BIMInspect — IFC Writer
Writes damage detection results into an IFC file as IfcAnnotation elements
with a Pset_DamageInspection property set.

Supports two modes:
  1. Template mode  — opens an existing IFC from ifc/templates/ and appends to it.
  2. Scratch mode   — creates a minimal IFC4 file from scratch when no template is given.

Output is always written to ifc/output/ (templates are never modified in place).

Pset_DamageInspection properties written per annotation:
  DamageClass          IfcLabel     e.g. "crack"
  Confidence           IfcReal      0.0 – 1.0
  ImagePath            IfcURIReference
  BBoxPixelX1          IfcInteger   pixel coords (top-left)
  BBoxPixelY1          IfcInteger
  BBoxPixelX2          IfcInteger   pixel coords (bottom-right)
  BBoxPixelY2          IfcInteger
  BBoxNormX1           IfcReal      normalised 0-1 coords
  BBoxNormY1           IfcReal
  BBoxNormX2           IfcReal
  BBoxNormY2           IfcReal
  ImageWidth           IfcInteger   source image dimensions
  ImageHeight          IfcInteger
  InspectionDate       IfcDate      ISO-8601 date of detection
  InspectionTool       IfcLabel     "BIMInspect / YOLOv8n-cls"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "ifc" / "templates"
OUTPUT_DIR    = ROOT / "ifc" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOOL_NAME  = "BIMInspect / YOLOv8n-cls"
IFC_SCHEMA = "IFC4"


# ── Input dataclass ────────────────────────────────────────────────────────────

@dataclass
class DamageRecord:
    """
    Damage detection result ready for IFC export.
    Mirrors the fields from src.detection.detector.DetectionResult
    but is decoupled so ifc_writer has no hard dependency on detector.py.
    """
    damage_class:       str                                      # "crack" | "no_crack"
    confidence:         float                                    # 0.0 – 1.0
    image_path:         str           = ""
    bbox:               Optional[tuple[int, int, int, int]] = None   # px (x1,y1,x2,y2)
    bbox_normalized:    Optional[tuple[float, float, float, float]] = None
    img_width:          int           = 0
    img_height:         int           = 0
    storey_name:        str           = "Ground Floor"           # target storey in IFC
    inspection_date:    Optional[date] = None                    # defaults to today


def from_detection_result(det) -> DamageRecord:
    """
    Convenience converter from src.detection.detector.DetectionResult
    to DamageRecord without creating a hard import dependency.
    """
    return DamageRecord(
        damage_class    = det.damage_class,
        confidence      = det.confidence,
        image_path      = str(det.image_path),
        bbox            = det.bbox,
        bbox_normalized = det.bbox_normalized,
        img_width       = det.img_width,
        img_height      = det.img_height,
    )


# ── IFC Writer ─────────────────────────────────────────────────────────────────

class IFCWriter:
    """
    Appends IfcAnnotation damage records to an IFC file.

    Usage — from a template:
        writer = IFCWriter(template="building.ifc")
        writer.write(record)
        path = writer.save()

    Usage — from scratch (no existing BIM model):
        writer = IFCWriter()
        writer.write(record1)
        writer.write(record2)
        path = writer.save("inspection_2024.ifc")
    """

    def __init__(self, template: Optional[str | Path] = None) -> None:
        if template:
            template = Path(template)
            # Resolve relative paths against the templates directory
            if not template.is_absolute():
                template = TEMPLATES_DIR / template
            if not template.exists():
                raise FileNotFoundError(f"IFC template not found: {template}")
            self.model = ifcopenshell.open(str(template))
            print(f"Opened template: {template}  (schema: {self.model.schema})")
        else:
            self.model = self._create_minimal_ifc()
            print("Created new IFC4 file from scratch.")

        self._annotation_count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def write(self, record: DamageRecord) -> ifcopenshell.entity_instance:
        """
        Write one damage detection result as an IfcAnnotation and return the entity.

        Args:
            record: DamageRecord with detection results.

        Returns:
            The created IfcAnnotation entity.
        """
        self._annotation_count += 1
        storey = self._get_or_create_storey(record.storey_name)

        # ── Annotation entity ──────────────────────────────────────────────────
        label = f"{record.damage_class.replace('_', ' ').title()} #{self._annotation_count}"
        annotation = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class   = "IfcAnnotation",
            name        = label,
        )
        annotation.Description = (
            f"{record.damage_class} detected with "
            f"{record.confidence * 100:.1f}% confidence"
        )
        annotation.ObjectType = record.damage_class

        # ── Contain inside the target storey ───────────────────────────────────
        ifcopenshell.api.run(
            "spatial.assign_container",
            self.model,
            relating_structure = storey,
            products           = [annotation],
        )

        # ── Pset_DamageInspection ──────────────────────────────────────────────
        pset = ifcopenshell.api.run(
            "pset.add_pset",
            self.model,
            product = annotation,
            name    = "Pset_DamageInspection",
        )

        inspection_date = record.inspection_date or date.today()
        props: dict = {
            "DamageClass":   record.damage_class,
            "Confidence":    round(record.confidence, 6),
            "ImagePath":     record.image_path or "",
            "ImageWidth":    record.img_width,
            "ImageHeight":   record.img_height,
            "InspectionDate": str(inspection_date),
            "InspectionTool": TOOL_NAME,
        }

        # Pixel bounding box
        if record.bbox:
            x1, y1, x2, y2 = record.bbox
            props.update({
                "BBoxPixelX1": x1,
                "BBoxPixelY1": y1,
                "BBoxPixelX2": x2,
                "BBoxPixelY2": y2,
            })

        # Normalised bounding box
        if record.bbox_normalized:
            nx1, ny1, nx2, ny2 = record.bbox_normalized
            props.update({
                "BBoxNormX1": round(nx1, 6),
                "BBoxNormY1": round(ny1, 6),
                "BBoxNormX2": round(nx2, 6),
                "BBoxNormY2": round(ny2, 6),
            })

        ifcopenshell.api.run(
            "pset.edit_pset",
            self.model,
            pset       = pset,
            properties = props,
        )

        print(
            f"  [+] {label}  confidence={record.confidence:.4f}"
            + (f"  bbox={record.bbox}" if record.bbox else "  bbox=None")
        )
        return annotation

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Write the IFC file to ifc/output/ and return the path.

        Args:
            filename: Output filename. Defaults to
                      biminspect_<YYYYMMDD_HHMMSS>.ifc
        """
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"biminspect_{ts}.ifc"

        out_path = OUTPUT_DIR / filename
        self.model.write(str(out_path))
        print(f"Saved: {out_path}  ({self._annotation_count} annotation(s))")
        return out_path

    # ── IFC helpers ────────────────────────────────────────────────────────────

    def _create_minimal_ifc(self) -> ifcopenshell.file:
        """Build the minimum valid IFC4 structure required to host annotations."""
        model = ifcopenshell.file(schema=IFC_SCHEMA)

        project = ifcopenshell.api.run(
            "root.create_entity", model,
            ifc_class = "IfcProject",
            name      = "BIMInspect Project",
        )
        ifcopenshell.api.run(
            "unit.assign_unit", model,
            length = {"is_metric": True, "raw": "METRES"},
        )
        ifcopenshell.api.run(
            "context.add_context", model,
            context_type = "Model",
        )

        site = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class="IfcSite", name="Site"
        )
        building = ifcopenshell.api.run(
            "root.create_entity", model, ifc_class="IfcBuilding", name="Building"
        )
        storey = ifcopenshell.api.run(
            "root.create_entity", model,
            ifc_class = "IfcBuildingStorey",
            name      = "Ground Floor",
        )

        ifcopenshell.api.run(
            "aggregate.assign_object", model,
            relating_object = project,
            products        = [site],
        )
        ifcopenshell.api.run(
            "aggregate.assign_object", model,
            relating_object = site,
            products        = [building],
        )
        ifcopenshell.api.run(
            "aggregate.assign_object", model,
            relating_object = building,
            products        = [storey],
        )
        return model

    def _get_or_create_storey(self, name: str) -> ifcopenshell.entity_instance:
        """Return the first IfcBuildingStorey matching name, or create one."""
        for storey in self.model.by_type("IfcBuildingStorey"):
            if storey.Name == name:
                return storey

        # Storey not found — attach a new one to the first building in the model
        buildings = self.model.by_type("IfcBuilding")
        building = buildings[0] if buildings else self._ensure_building()

        new_storey = ifcopenshell.api.run(
            "root.create_entity", self.model,
            ifc_class = "IfcBuildingStorey",
            name      = name,
        )
        ifcopenshell.api.run(
            "aggregate.assign_object", self.model,
            relating_object = building,
            products        = [new_storey],
        )
        print(f"  Created new storey: '{name}'")
        return new_storey

    def _ensure_building(self) -> ifcopenshell.entity_instance:
        """Last-resort: create a building hierarchy if the IFC has none."""
        sites = self.model.by_type("IfcSite")
        site = sites[0] if sites else ifcopenshell.api.run(
            "root.create_entity", self.model, ifc_class="IfcSite", name="Site"
        )
        building = ifcopenshell.api.run(
            "root.create_entity", self.model, ifc_class="IfcBuilding", name="Building"
        )
        ifcopenshell.api.run(
            "aggregate.assign_object", self.model,
            relating_object = site,
            products        = [building],
        )
        return building


# ── Convenience function ───────────────────────────────────────────────────────

def write_detection_to_ifc(
    detection_result,
    template:    Optional[str | Path] = None,
    output_name: Optional[str]        = None,
    storey_name: str                  = "Ground Floor",
) -> Path:
    """
    One-shot helper: take a DetectionResult, write it to IFC, return output path.

    Args:
        detection_result: DetectionResult from src.detection.detector.
        template:         Optional IFC template filename (relative to ifc/templates/).
        output_name:      Optional output filename (relative to ifc/output/).
        storey_name:      Target building storey name in the IFC hierarchy.

    Returns:
        Path to the saved IFC file.
    """
    record = from_detection_result(detection_result)
    record.storey_name = storey_name

    writer = IFCWriter(template=template)
    writer.write(record)
    return writer.save(output_name)


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick self-test: write a synthetic detection result to a new IFC file
    from datetime import date

    print("=== BIMInspect IFC Writer — self-test ===\n")

    record = DamageRecord(
        damage_class    = "crack",
        confidence      = 0.9998,
        image_path      = "data/raw/Positive/00001.jpg",
        bbox            = (126, 182, 166, 227),
        bbox_normalized = (0.5551, 0.8018, 0.7313, 1.0),
        img_width       = 227,
        img_height      = 227,
        storey_name     = "Ground Floor",
        inspection_date = date.today(),
    )

    writer = IFCWriter()          # create from scratch
    ann    = writer.write(record)
    path   = writer.save("self_test.ifc")

    # Verify by re-opening and reading back
    print("\n=== Verification (re-open and read back) ===")
    check = ifcopenshell.open(str(path))
    for annotation in check.by_type("IfcAnnotation"):
        print(f"\nAnnotation : {annotation.Name}")
        print(f"Description: {annotation.Description}")
        for rel in check.by_type("IfcRelDefinesByProperties"):
            if annotation in rel.RelatedObjects:
                pset = rel.RelatingPropertyDefinition
                print(f"Pset       : {pset.Name}")
                for prop in pset.HasProperties:
                    print(f"  {prop.Name:20s} = {prop.NominalValue}")

# Automated Damage Detection — Project Context

## What this project does

Automatically detects structural damage (cracks, spalling, corrosion, delamination, etc.) in construction photos and videos, maps each detection to real-world building coordinates, and writes the results back into an IFC (Industry Foundation Classes) BIM model so damage is queryable, visualisable, and exportable in any BIM authoring tool (Revit, ArchiCAD, Solibri, etc.).

## Goals

1. **Automate damage surveys** — replace manual walkthroughs with camera footage + AI.
2. **Geo-reference every defect** — convert pixel bounding boxes into 3-D building coordinates.
3. **Enrich the BIM model** — attach damage severity, type, area, and photos as IFC properties.
4. **Generate inspection reports** — produce PDF/HTML summaries directly from IFC data.

---

## Pipeline overview

```
[Input images / video frames]
        │
        ▼
┌───────────────────┐
│  1. YOLOv8        │  src/detection/
│  Inference        │  • Loads a fine-tuned YOLOv8 model from models/weights/
│                   │  • Outputs bounding boxes, class labels, confidence scores
└────────┬──────────┘
         │  detections (pixel coords + metadata)
         ▼
┌───────────────────┐
│  2. Coordinate    │  src/pipeline/
│  Mapping          │  • Camera intrinsics / extrinsics → ray casting
│                   │  • OR: known reference markers → homography
│                   │  • Pixel bbox  →  3-D world XYZ  →  IFC local coords
└────────┬──────────┘
         │  damage objects with 3-D position + extent
         ▼
┌───────────────────┐
│  3. IFC Export    │  src/bim/
│                   │  • Opens template IFC from ifc/templates/
│                   │  • Creates IfcBuildingElement or IfcAnnotation per defect
│                   │  • Attaches Pset_DamageProperties (type, severity, area…)
│                   │  • Saves enriched model to ifc/output/
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. Reporting     │  src/utils/  +  results/
│                   │  • JSON / CSV detections → results/detections/
│                   │  • Annotated images      → results/visualizations/
│                   │  • PDF / HTML reports    → results/reports/
└───────────────────┘
```

---

## Folder structure

```
Automated_Damage_Detection/
│
├── CLAUDE.md               ← you are here
├── requirements.txt        ← all Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                ← original, unmodified images and video frames
│   │   ├── Positive/       ← 20,000 crack images (READY — arunrk7/surface-crack-detection)
│   │   ├── Negative/       ← 20,000 no-crack images (READY)
│   │   └── surface-crack-detection.zip
│   ├── annotated/          ← labelled data in YOLO format (.txt label files)
│   ├── processed/          ← resized / augmented images ready for training
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── weights/            ← .pt model files (YOLOv8n/s/m/l/x, fine-tuned)
│   ├── configs/            ← dataset YAML files, hyperparameter configs
│   └── exports/            ← ONNX, TensorRT, CoreML exported models
│
├── src/
│   ├── detection/          ← YOLOv8 training, inference, post-processing
│   ├── bim/                ← IFC read/write, property set creation
│   ├── pipeline/           ← end-to-end orchestration, coordinate mapping
│   └── utils/              ← shared helpers (logging, geometry, I/O, reporting)
│
├── tests/                  ← pytest unit + integration tests
│
├── ifc/
│   ├── templates/          ← base IFC files representing the building model
│   └── output/             ← enriched IFC files after damage annotation
│
└── results/
    ├── detections/         ← raw JSON / CSV output per image or session
    ├── reports/            ← final PDF / HTML inspection reports
    └── visualizations/     ← images with bounding boxes and labels overlaid
```

---

## Key dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 training and inference |
| `ifcopenshell` | Read and write IFC / BIM files |
| `torch` / `torchvision` | Deep learning backend |
| `opencv-python` | Image I/O, homography, drawing |
| `shapely` | 2-D / 3-D geometric operations |
| `pandas` | Tabular detection results |
| `plotly` / `matplotlib` | Visualisation and reporting |
| `pytest` | Unit and integration testing |

Install everything with:
```bash
pip install -r requirements.txt
```

---

## Damage classes (planned)

- `crack` — hairline to structural cracks
- `spalling` — concrete surface loss
- `corrosion` — rebar / steel rust staining
- `delamination` — surface layer separation
- `efflorescence` — salt deposits / moisture ingress
- `void` — missing material / holes

---

## Dataset status

| Split | Location | Count | Status |
|---|---|---|---|
| Raw — crack | `data/raw/Positive/` | 20,000 | Ready |
| Raw — no crack | `data/raw/Negative/` | 20,000 | Ready |
| Annotated (YOLO) | `data/annotated/` | — | Pending |
| Train / Val / Test splits | `data/splits/` | — | Pending |

Downloaded via `tests/download_data.py` from Kaggle dataset `arunrk7/surface-crack-detection`.
Images are 227×227 px RGB JPEGs. Next step: convert to YOLO bounding-box format and split.

---

## Development notes

- Model weights go in `models/weights/` and are excluded from git (large files).
- IFC templates in `ifc/templates/` represent the as-built building; never overwrite them — always write enriched copies to `ifc/output/`.
- Coordinate mapping strategy depends on available survey data: homography for flat surfaces with reference markers, full photogrammetry pipeline for complex geometry.
- All source modules are under `src/` as proper Python packages (`__init__.py` present in each).
- Run tests from the project root with `pytest tests/`.

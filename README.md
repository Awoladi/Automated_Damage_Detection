<p align="center">
  <img src="assets/logo.png" alt="BIMInspect Logo" width="320"/>
</p>

<h1 align="center">BIMInspect</h1>

<p align="center">
  <strong>AI-powered structural damage detection that writes directly into your BIM model.</strong><br/>
  YOLOv8 · IFC / ifcopenshell · PyTorch · OpenCV
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-blue?logo=python" />
  <img src="https://img.shields.io/badge/torch-2.10.0%2Bcu126-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-purple" />
  <img src="https://img.shields.io/badge/IFC-ifcopenshell-0078d7" />
  <img src="https://img.shields.io/badge/GPU-RTX%203070-76b900?logo=nvidia" />
</p>

---

## What is BIMInspect?

**BIMInspect** automates structural damage surveys. Point a camera at a building, run the pipeline, and every crack, spalling patch, or corrosion stain is:

1. **Detected** by a fine-tuned YOLOv8 model
2. **Geo-referenced** — pixel bounding boxes converted to 3-D building coordinates
3. **Written into the BIM model** as IFC objects with full property sets
4. **Reported** as PDF / HTML inspection summaries

No more manual walkthroughs. No more spreadsheets. Every defect is queryable in Revit, ArchiCAD, or Solibri the moment the pipeline finishes.

---

## Pipeline

```
[Input images / video frames]
        │
        ▼
┌───────────────────┐
│  1. YOLOv8        │  src/detection/
│  Inference        │  • Fine-tuned on 40,000 crack / no-crack images
│                   │  • Outputs bounding boxes, class labels, confidence
└────────┬──────────┘
         │  detections (pixel coords + metadata)
         ▼
┌───────────────────┐
│  2. Coordinate    │  src/pipeline/
│  Mapping          │  • Camera intrinsics / extrinsics → ray casting
│                   │  • OR: reference markers → homography
│                   │  • Pixel bbox → 3-D world XYZ → IFC local coords
└────────┬──────────┘
         │  damage objects with 3-D position + extent
         ▼
┌───────────────────┐
│  3. IFC Export    │  src/bim/
│                   │  • Opens template IFC (as-built model)
│                   │  • Creates IfcAnnotation per defect
│                   │  • Attaches Pset_DamageProperties
│                   │  • Saves enriched model to ifc/output/
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. Reporting     │  src/utils/ + results/
│                   │  • JSON / CSV → results/detections/
│                   │  • Annotated images → results/visualizations/
│                   │  • PDF / HTML → results/reports/
└───────────────────┘
```

---

## Damage Classes

| Class | Description |
|---|---|
| `crack` | Hairline to structural cracks |
| `spalling` | Concrete surface loss |
| `corrosion` | Rebar / steel rust staining |
| `delamination` | Surface layer separation |
| `efflorescence` | Salt deposits / moisture ingress |
| `void` | Missing material / holes |

---

## Project Structure

```
BIMInspect/
├── assets/                 ← logo and static resources
├── data/
│   ├── raw/
│   │   ├── Positive/       ← 20,000 crack images  ✅
│   │   └── Negative/       ← 20,000 no-crack images  ✅
│   ├── annotated/          ← YOLO-format labels
│   ├── processed/          ← augmented training images
│   └── splits/train|val|test/
├── models/
│   ├── weights/            ← .pt model files (git-ignored)
│   ├── configs/            ← dataset YAML + hyperparameters
│   └── exports/            ← ONNX / TensorRT / CoreML
├── src/
│   ├── detection/          ← YOLOv8 training & inference
│   ├── bim/                ← IFC read/write, property sets
│   ├── pipeline/           ← end-to-end orchestration
│   └── utils/              ← logging, geometry, reporting
├── tests/                  ← pytest unit + integration tests
├── ifc/
│   ├── templates/          ← as-built IFC models (read-only)
│   └── output/             ← enriched IFC after annotation
└── results/
    ├── detections/         ← JSON / CSV per session
    ├── visualizations/     ← annotated images
    └── reports/            ← PDF / HTML summaries
```

---

## Quickstart

### 1. Clone & create environment
```bash
git clone https://github.com/Awoladi/Automated_Damage_Detection.git
cd Automated_Damage_Detection
py -3 -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install GPU PyTorch (CUDA 12.6)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Download dataset
```bash
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json
python tests/download_data.py
```

### 5. Verify GPU
```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))
```

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥ 8.2 | YOLOv8 training & inference |
| `ifcopenshell` | ≥ 0.7 | IFC / BIM read & write |
| `torch` + `torchvision` | 2.10 + cu126 | Deep learning (GPU) |
| `opencv-python` | ≥ 4.9 | Image I/O, homography |
| `shapely` | ≥ 2.0 | 2-D / 3-D geometry |
| `pandas` | ≥ 2.2 | Tabular results |
| `plotly` / `matplotlib` | ≥ 5.20 / 3.8 | Visualisation & reporting |
| `kaggle` | ≥ 2.0 | Dataset download |
| `pytest` | ≥ 8.0 | Testing |

---

## Dataset

| Split | Location | Images | Status |
|---|---|---|---|
| Crack (Positive) | `data/raw/Positive/` | 20,000 | ✅ Ready |
| No-crack (Negative) | `data/raw/Negative/` | 20,000 | ✅ Ready |
| YOLO annotations | `data/annotated/` | — | Pending |
| Train / Val / Test | `data/splits/` | — | Pending |

Source: [arunrk7/surface-crack-detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) — 227×227 px RGB JPEGs.

---

## License

MIT — see [LICENSE](LICENSE) for details.

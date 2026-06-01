<p align="center">
  <img src="assets/logo.png" alt="BIMInspect Logo" width="320"/>
</p>

<h1 align="center">BIMInspect</h1>

<p align="center">
  <strong>AI-powered structural damage detection that writes directly into your BIM model.</strong><br/>
  YOLOv8 · IFC / ifcopenshell · PyTorch · Streamlit · OpenCV
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-blue?logo=python" />
  <img src="https://img.shields.io/badge/torch-2.10.0%2Bcu126-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-purple" />
  <img src="https://img.shields.io/badge/IFC-ifcopenshell-0078d7" />
  <img src="https://img.shields.io/badge/GPU-RTX%203070-76b900?logo=nvidia" />
  <img src="https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit" />
</p>

---

## What is BIMInspect?

**BIMInspect** automates structural damage surveys. Point a camera at a building, upload the photo to the dashboard, and every defect is:

1. **Detected** by a fine-tuned YOLOv8m object detection model across 5 damage classes
2. **Localised** with a tight bounding box around the exact defect area
3. **Written into the BIM model** as an `IfcAnnotation` with `Pset_DamageInspection`
4. **Exported** as a ready-to-open IFC file — viewable in Revit, ArchiCAD, or Solibri

---

## Dashboard

```bash
venv\Scripts\python -m streamlit run app.py
```

Upload a photo → damage class, confidence score, annotated bounding box → download the IFC file.

---

## Pipeline

```
[Input image]
      │
      ▼
┌─────────────────────┐
│  1. YOLOv8m         │  src/detection/detector.py
│  Tiled Inference    │  • Slides 640×640 window over full-res image
│                     │  • NMS across tiles → final detections
└──────────┬──────────┘
           │  class · confidence · bbox (x1,y1,x2,y2)
           ▼
┌─────────────────────┐
│  2. IFC Export      │  src/bim/ifc_writer.py
│                     │  • Creates IfcAnnotation per detection
│                     │  • Pset_DamageInspection: class, confidence,
│                     │    bbox px + normalised, date, tool
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Streamlit UI    │  app.py
│                     │  • Annotated image with all detections
│                     │  • Download IFC file
└─────────────────────┘
```

---

## Model Status

| Version | Dataset | mAP@50 | Notes |
|---|---|---|---|
| **v12 — manual labels** | 2,041 hand-labeled | *in training* | All 5 classes manually annotated from CODEBRIM |
| v10 — auto CODEBRIM | 13,057 tiles | **57.7%** | Best so far — current `best_detection.pt` |
| v11 — CODEBRIM cracks | 12,377 tiles | 51.4% | Added CODEBRIM crack tiles; hurt performance |
| v9 — tiled | 8,800 tiles | 57.8% | Full-res tiling, YOLOv8s |
| v7 — real bboxes | 9,168 images | 57.5% | First tight CODEBRIM bboxes |
| v4 — crack only | 4,000 images | 99.4% | Single class, manual labels |

---

## Damage Classes

| ID | Class | v12 Training Images |
|---|---|---|
| 0 | `crack` | 471 manually labeled |
| 1 | `spallation` | 291 manually labeled |
| 2 | `efflorescence` | 389 manually labeled |
| 3 | `exposed_bars` | 429 manually labeled |
| 4 | `corrosion` | 461 manually labeled |

All from CODEBRIM building facade tiles — single consistent visual domain.

---

## Project Structure

```
BIMInspect/
├── app.py                          ← Streamlit dashboard
├── train.py                        ← YOLOv8m training script
├── start_labelstudio.bat           ← Launch Label Studio for annotation
├── assets/                         ← logo
├── data/
│   ├── raw/
│   │   └── CODEBRIM_original_images.zip  ← 7.8 GB (Zenodo)
│   ├── to_label/                   ← 500 CODEBRIM tiles per class (for labeling)
│   └── expanded_multiclass/        ← built training dataset (generated)
├── models/
│   ├── weights/                    ← .pt model files (git-ignored)
│   │   ├── best_detection.pt       ← production model
│   │   └── best.pt                 ← legacy binary classifier (fallback)
│   ├── configs/
│   │   └── dataset_multiclass.yaml ← YOLO training config
│   └── exports/                    ← ONNX / TensorRT / CoreML
├── src/
│   ├── detection/
│   │   ├── detector.py                 ← DamageDetector (tiled inference + NMS)
│   │   ├── extract_for_labeling.py     ← Extract CODEBRIM tiles for manual labeling
│   │   └── build_dataset.py            ← Build dataset from Label Studio exports
│   ├── bim/
│   │   └── ifc_writer.py               ← IFCWriter + Pset_DamageInspection
│   └── pipeline/
│       └── pipeline.py                 ← end-to-end orchestration
├── ifc/
│   ├── templates/                  ← as-built IFC models (read-only)
│   └── output/                     ← enriched IFC after annotation
└── results/
    ├── detections/                 ← JSON / CSV per session
    ├── visualizations/             ← annotated images
    └── reports/                    ← PDF / HTML summaries
```

---

## Quickstart

### 1. Clone & environment
```bash
git clone https://github.com/Awoladi/Automated_Damage_Detection.git
cd Automated_Damage_Detection
py -3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 2. Download CODEBRIM
Download `CODEBRIM_original_images.zip` from [Zenodo record 2620293](https://zenodo.org/records/2620293) and place at `data/raw/CODEBRIM_original_images.zip`.

### 3. Label data
```bash
# Extract 500 representative tiles per class
venv\Scripts\python src/detection/extract_for_labeling.py

# Start Label Studio and annotate data/to_label/<class>/ folders
.\start_labelstudio.bat

# After exporting YOLO zips from Label Studio to Downloads/:
venv\Scripts\python src/detection/build_dataset.py
```

### 4. Train
```bash
venv\Scripts\python train.py
```

### 5. Run dashboard
```bash
venv\Scripts\python -m streamlit run app.py
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 training & inference |
| `ifcopenshell` | IFC / BIM read & write |
| `torch` + `torchvision` | Deep learning (GPU) |
| `opencv-python` | Image I/O and drawing |
| `streamlit` | Web dashboard |

---

## Dataset Sources

| Source | License |
|---|---|
| [CODEBRIM](https://zenodo.org/records/2620293) — building facade damage photos with bbox annotations | CC BY-NC 4.0 |

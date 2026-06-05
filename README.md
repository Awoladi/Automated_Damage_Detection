<p align="center">
  <img src="assets/logo.png" alt="BIMInspect Logo" width="320"/>
</p>

<h1 align="center">BIMInspect</h1>

<p align="center">
  <strong>AI-powered structural damage detection that writes directly into your BIM model.</strong><br/>
  YOLO11 · IFC / ifcopenshell · PyTorch · Streamlit · OpenCV
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-blue?logo=python" />
  <img src="https://img.shields.io/badge/torch-2.10.0%2Bcu126-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/YOLO11-ultralytics-purple" />
  <img src="https://img.shields.io/badge/IFC-ifcopenshell-0078d7" />
  <img src="https://img.shields.io/badge/GPU-RTX%203070-76b900?logo=nvidia" />
  <img src="https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit" />
</p>

---

## What is BIMInspect?

**BIMInspect** automates structural damage surveys. Point a camera at a building, upload the photo to the dashboard, and every defect is:

1. **Detected** by a fine-tuned YOLO11m object detection model across 5 damage classes
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
│  1. YOLO11m         │  src/detection/detector.py
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
| **v13 — combined** | Manual + CODEBRIM + Roboflow | *in training* | YOLO11m, copy_paste, 200 epochs |
| v12 — manual labels | 2,041 hand-labeled | 33.0% | All 5 classes manually annotated from CODEBRIM |
| v10 — auto CODEBRIM | 13,057 tiles | **57.7%** | Best so far — current `best_detection.pt` |
| v11 — CODEBRIM cracks | 12,377 tiles | 51.4% | Added CODEBRIM crack tiles; hurt performance |
| v9 — tiled | 8,800 tiles | 57.8% | Full-res tiling, YOLOv8s |
| v7 — real bboxes | 9,168 images | 57.5% | First tight CODEBRIM bboxes |
| v4 — crack only | 4,000 images | 99.4% | Single class, manual labels |

---

## Damage Classes

| ID | Class | v13 Sources |
|---|---|---|
| 0 | `crack` | Manual + CODEBRIM + Roboflow (Concrete 5Nov) |
| 1 | `spallation` | Manual + CODEBRIM + Roboflow (Spalling-Det, Concrete 5Nov) |
| 2 | `efflorescence` | Manual + CODEBRIM + Roboflow (Efflorescence-Det, Concrete 5Nov) |
| 3 | `exposed_bars` | Manual + CODEBRIM + Roboflow (Exposure-Det, Concrete 5Nov) |
| 4 | `corrosion` | Manual + CODEBRIM + Roboflow (corrosion, Concrete 5Nov) |

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

### 3. Prepare dataset
```bash
# Extract CODEBRIM zip to data/raw/CODEBRIM_original_extracted/
venv\Scripts\python src/detection/extract_codebrim.py

# (Optional) Label additional images in Label Studio:
#   1. venv\Scripts\python src/detection/extract_for_labeling.py
#   2. .\start_labelstudio.bat  -> annotate -> export YOLO zip to Downloads/

# Build combined training dataset (Manual + CODEBRIM + Roboflow)
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
| `ultralytics` | YOLO11 training & inference |
| `ifcopenshell` | IFC / BIM read & write |
| `torch` + `torchvision` | Deep learning (GPU) |
| `opencv-python` | Image I/O and drawing |
| `streamlit` | Web dashboard |

---

## Dataset Sources

| Source | License |
|---|---|
| [CODEBRIM](https://zenodo.org/records/2620293) — building facade damage photos with bbox annotations | CC BY-NC 4.0 |
| Manual labels — 2,041 images annotated in Label Studio from CODEBRIM tiles | — |
| [Roboflow: Concrete 5November](https://universe.roboflow.com) — 7,443 images, 5 damage classes | CC BY 4.0 |
| [Roboflow: corrosion](https://universe.roboflow.com) — 5,473 images, 3 severity levels | CC BY 4.0 |
| [Roboflow: Efflorescence-Det](https://universe.roboflow.com) — 1,011 images | CC BY 4.0 |
| [Roboflow: Exposure-Det](https://universe.roboflow.com) — 1,032 images | CC BY 4.0 |
| [Roboflow: Spalling-Det](https://universe.roboflow.com) — 1,009 images | CC BY 4.0 |

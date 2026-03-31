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

**BIMInspect** automates structural damage surveys. Point a camera at a building, upload the photo to the dashboard, and every crack is:

1. **Detected** by a fine-tuned YOLOv8n object detection model
2. **Localised** with a tight bounding box directly from the network
3. **Written into the BIM model** as an `IfcAnnotation` with `Pset_DamageInspection`
4. **Exported** as a ready-to-open IFC file — viewable in Revit, ArchiCAD, or Solibri

No more manual walkthroughs. No more spreadsheets.

---

## Dashboard

Upload a photo → get the damage class, confidence score, and annotated bounding box → download the IFC file — all in one screen.

```bash
# Activate venv, then:
PYTHONPATH=. venv\Scripts\streamlit run app.py
```

---

## Pipeline

```
[Input image]
      │
      ▼
┌─────────────────────┐
│  1. YOLOv8n         │  src/detection/detector.py
│  Object Detection   │  • best_detection.pt  (50 epochs, mAP@50 = 60.8%)
│                     │  • Native bboxes — no Grad-CAM needed
│                     │  • Fallback: YOLOv8n-cls + Grad-CAM (best.pt)
└──────────┬──────────┘
           │  class · confidence · bbox (x1,y1,x2,y2)
           ▼
┌─────────────────────┐
│  2. IFC Export      │  src/bim/ifc_writer.py
│                     │  • Opens template or creates new IFC4 file
│                     │  • Creates IfcAnnotation per detection
│                     │  • Pset_DamageInspection: class, confidence,
│                     │    bbox px + normalised, image path, date, tool
│                     │  • Saves to ifc/output/
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Streamlit UI    │  app.py
│                     │  • Upload photo
│                     │  • View annotated image + metrics
│                     │  • Download IFC file
└─────────────────────┘
```

---

## Model Status

| Model | File | Epochs | mAP@50 | Precision | Recall | Mode |
|---|---|---|---|---|---|---|
| YOLOv8n detector | `best_detection.pt` | 50 | **60.8%** | 71.5% | 72.5% | Detection ✅ |
| YOLOv8n-cls | `best.pt` | 20 | — | — | **99.8% top-1** | Classification (fallback) |

> Labels for the detection model were auto-generated via Grad-CAM from the classifier.
> Manual annotation with Label Studio is in progress to improve bbox tightness.

---

## Damage Classes

| Class | Description |
|---|---|
| `crack` | Hairline to structural cracks |
| `spalling` | Concrete surface loss *(planned)* |
| `corrosion` | Rebar / steel rust staining *(planned)* |
| `delamination` | Surface layer separation *(planned)* |
| `efflorescence` | Salt deposits / moisture ingress *(planned)* |
| `void` | Missing material / holes *(planned)* |

---

## Project Structure

```
BIMInspect/
├── app.py                  ← Streamlit dashboard
├── assets/                 ← logo and static resources
├── data/
│   ├── raw/
│   │   ├── Positive/       ← 20,000 crack images  ✅
│   │   └── Negative/       ← 20,000 no-crack images  ✅
│   ├── detection/          ← YOLO detection dataset (auto-labeled)  ✅
│   │   ├── train/          ← 28,000 images + labels
│   │   ├── val/            ← 8,000 images + labels
│   │   └── test/           ← 4,000 images + labels
│   └── labeling/           ← Label Studio manual annotation sample
├── models/
│   ├── weights/            ← .pt model files (git-ignored)
│   ├── configs/            ← dataset YAML + hyperparameters
│   └── exports/            ← ONNX / TensorRT / CoreML
├── src/
│   ├── detection/          ← YOLOv8 training, inference, label generation
│   │   ├── detector.py     ← DamageDetector (detection + cls fallback)
│   │   ├── train.py        ← classifier training
│   │   ├── train_detection.py  ← detector training
│   │   └── generate_labels.py  ← Grad-CAM auto-labeler
│   ├── bim/
│   │   └── ifc_writer.py   ← IFCWriter + Pset_DamageInspection
│   ├── pipeline/
│   │   └── pipeline.py     ← end-to-end orchestration
│   └── utils/              ← shared helpers
├── tests/
│   └── download_data.py    ← Kaggle dataset downloader
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

### 5. Train models
```bash
# Step 1 — classifier (used to generate detection labels)
PYTHONPATH=. venv\Scripts\python src/detection/train.py

# Step 2 — auto-generate YOLO bounding box labels via Grad-CAM
PYTHONPATH=. venv\Scripts\python src/detection/generate_labels.py

# Step 3 — train YOLOv8n object detector
PYTHONPATH=. venv\Scripts\python src/detection/train_detection.py
```

### 6. Launch dashboard
```bash
PYTHONPATH=. venv\Scripts\streamlit run app.py
```

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥ 8.2 | YOLOv8 training & inference |
| `ifcopenshell` | ≥ 0.7 | IFC / BIM read & write |
| `torch` + `torchvision` | 2.10 + cu126 | Deep learning (GPU) |
| `opencv-python` | ≥ 4.9 | Image I/O, homography, drawing |
| `streamlit` | ≥ 1.55 | Web dashboard |
| `shapely` | ≥ 2.0 | 2-D / 3-D geometry |
| `pandas` | ≥ 2.2 | Tabular results |
| `kaggle` | ≥ 2.0 | Dataset download |
| `label-studio` | ≥ 1.23 | Manual annotation |
| `pytest` | ≥ 8.0 | Testing |

---

## Dataset

| Split | Location | Images | Status |
|---|---|---|---|
| Crack (raw) | `data/raw/Positive/` | 20,000 | ✅ Ready |
| No-crack (raw) | `data/raw/Negative/` | 20,000 | ✅ Ready |
| Detection train | `data/detection/train/` | 28,000 | ✅ Auto-labeled |
| Detection val | `data/detection/val/` | 8,000 | ✅ Auto-labeled |
| Detection test | `data/detection/test/` | 4,000 | ✅ Auto-labeled |
| Manual labels | `data/labeling/` | 500 | 🔄 In progress |

Source: [arunrk7/surface-crack-detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) — 227×227 px RGB JPEGs.

---

## License

MIT — see [LICENSE](LICENSE) for details.

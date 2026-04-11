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

1. **Detected** by a fine-tuned YOLOv8n object detection model
2. **Localised** with a tight bounding box directly from the network
3. **Written into the BIM model** as an `IfcAnnotation` with `Pset_DamageInspection`
4. **Exported** as a ready-to-open IFC file — viewable in Revit, ArchiCAD, or Solibri

No more manual walkthroughs. No more spreadsheets.

---

## Dashboard

Upload a photo → get the damage class, confidence score, and annotated bounding box → download the IFC file — all in one screen.

```bash
venv\Scripts\python -m streamlit run app.py
```

---

## Pipeline

```
[Input image]
      │
      ▼
┌─────────────────────┐
│  1. YOLOv8n         │  src/detection/detector.py
│  Object Detection   │  • best_detection.pt  (150 epochs, mAP@50 = 99.4%)
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

| Model | File | Epochs | mAP@50 | Precision | Recall | Notes |
|---|---|---|---|---|---|---|
| YOLOv8n detector | `best_detection.pt` | 150 | **99.4%** | 99.1% | 99.2% | Trained on 4,000 clean manual labels |
| YOLOv8n-cls | `best.pt` | 20 | — | — | 99.8% top-1 | Classification fallback |

Training uses 500 manually annotated images expanded to 4,000 via 8-way geometric augmentation (flip H/V, rotate 90/180/270, transpose, transverse).

---

## Damage Classes

| Class | Training Data | Status |
|---|---|---|
| `crack` | 4,000 expanded manual labels | **Live — 99.4% mAP** |
| `spallation` | 500 images (CODEBRIM) | Labeling next |
| `corrosion` | 500 images (CODEBRIM) | Labeling next |
| `efflorescence` | 500 images (CODEBRIM) | Labeling next |
| `exposed_bars` | 500 images (CODEBRIM) | Labeling next |

---

## Project Structure

```
BIMInspect/
├── app.py                      ← Streamlit dashboard
├── train.py                    ← YOLOv8 detection training script
├── assets/                     ← logo and static resources
├── data/
│   ├── raw/
│   │   ├── Positive/           ← 20,000 crack images
│   │   ├── Negative/           ← 20,000 no-crack images
│   │   ├── spallation/         ← 500 images (CODEBRIM)
│   │   ├── corrosion/          ← 500 images (CODEBRIM)
│   │   ├── efflorescence/      ← 500 images (CODEBRIM)
│   │   └── exposed_bars/       ← 500 images (CODEBRIM)
│   ├── annotated/              ← 500 manually labeled crack images (YOLO format)
│   ├── expanded_manual/        ← 4,000 augmented training images
│   └── labeling/               ← Label Studio annotation working folder
├── models/
│   ├── weights/                ← .pt model files (git-ignored)
│   ├── configs/                ← dataset YAML configs
│   └── exports/                ← ONNX / TensorRT / CoreML
├── src/
│   ├── detection/
│   │   ├── detector.py         ← DamageDetector (detection + cls fallback)
│   │   ├── expand_manual_labels.py  ← 8-way geometric augmentation
│   │   └── download_codebrim.py     ← CODEBRIM dataset downloader
│   ├── bim/
│   │   └── ifc_writer.py       ← IFCWriter + Pset_DamageInspection
│   ├── pipeline/
│   │   └── pipeline.py         ← end-to-end orchestration
│   └── utils/                  ← shared helpers
├── tests/
│   └── download_data.py        ← Kaggle crack dataset downloader
├── ifc/
│   ├── templates/              ← as-built IFC models (read-only)
│   └── output/                 ← enriched IFC after annotation
└── results/
    ├── detections/             ← JSON / CSV per session
    ├── visualizations/         ← annotated images
    └── reports/                ← PDF / HTML summaries
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

### 4. Launch dashboard
```bash
venv\Scripts\python -m streamlit run app.py
```

### 5. Retrain the detector
```bash
# Expand manual labels (500 → 4,000 images)
venv\Scripts\python src/detection/expand_manual_labels.py

# Train
venv\Scripts\python train.py
```

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | >= 8.2 | YOLOv8 training & inference |
| `ifcopenshell` | >= 0.7 | IFC / BIM read & write |
| `torch` + `torchvision` | 2.10 + cu126 | Deep learning (GPU) |
| `opencv-python` | >= 4.9 | Image I/O and drawing |
| `streamlit` | >= 1.55 | Web dashboard |
| `shapely` | >= 2.0 | 2-D / 3-D geometry |
| `pandas` | >= 2.2 | Tabular results |
| `label-studio` | >= 1.23 | Manual annotation |
| `pytest` | >= 8.0 | Testing |

---

## Dataset Sources

| Class | Source | Images |
|---|---|---|
| crack / no-crack | [arunrk7/surface-crack-detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (Kaggle) | 40,000 |
| spallation, corrosion, efflorescence, exposed bars | [CODEBRIM](https://zenodo.org/records/2620293) (Zenodo, CC BY-NC 4.0) | 500 each |

---

## License

MIT — see [LICENSE](LICENSE) for details.

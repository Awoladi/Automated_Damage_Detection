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

1. **Detected** by a fine-tuned YOLOv8n object detection model across 5 damage classes
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
│  Object Detection   │  • best_detection.pt  (150 epochs, mAP@50 = 97.8%)
│                     │  • 5 damage classes
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
| YOLOv8n detector (5-class) | `best_detection.pt` | 75 (early stop) | **97.8%** | — | — | Trained on 12,496 images, 5 damage types |
| YOLOv8n detector (crack only) | *(archived)* | 150 | 99.4% | 99.1% | 99.2% | Superseded by 5-class model |
| YOLOv8n-cls (fallback) | `best.pt` | 20 | — | — | 99.8% top-1 | Binary crack/no-crack classifier |

---

## Damage Classes

All 5 classes are live in the current `best_detection.pt`:

| ID | Class | Training Images | Source |
|---|---|---|---|
| 0 | `crack` | 4,000 (500 manual × 8-way augment) | Hand-labeled in Label Studio |
| 1 | `spallation` | 4,000 (500 CODEBRIM crops × 8-way augment) | CODEBRIM classification dataset |
| 2 | `efflorescence` | 4,000 (500 CODEBRIM crops × 8-way augment) | CODEBRIM classification dataset |
| 3 | `exposed_bars` | ~712 (89 CODEBRIM crops × 8-way augment) | CODEBRIM classification dataset |
| 4 | `corrosion` | 4,000 (500 CODEBRIM crops × 8-way augment) | CODEBRIM classification dataset |

`exposed_bars` has fewer samples because CODEBRIM contains fewer single-class exposed-bar images.

---

## Project Structure

```
BIMInspect/
├── app.py                          ← Streamlit dashboard
├── train.py                        ← YOLOv8 training script (v7 fresh start)
├── assets/                         ← logo and static resources
├── data/
│   ├── raw/                        ← original images (git-ignored)
│   │   ├── Positive/               ← 20,000 crack images (Kaggle)
│   │   └── Negative/               ← 20,000 no-crack images (Kaggle)
│   ├── annotated/                  ← 500 manually labeled crack images (YOLO format)
│   ├── annotated_multiclass/       ← 500 CODEBRIM classification crops per class
│   ├── expanded_manual/            ← 4,000 augmented crack training images
│   └── expanded_multiclass/        ← 12,496 train + 2,192 val (all 5 classes)
├── models/
│   ├── weights/                    ← .pt model files (git-ignored)
│   │   ├── best_detection.pt       ← current production model (5-class, 97.8% mAP)
│   │   └── best.pt                 ← fallback classifier (crack/no-crack)
│   ├── configs/
│   │   └── dataset_multiclass.yaml ← dataset config for training
│   └── exports/                    ← ONNX / TensorRT / CoreML
├── src/
│   ├── detection/
│   │   ├── detector.py             ← DamageDetector (5-class detection + cls fallback)
│   │   ├── expand_manual_labels.py ← 8-way augmentation for crack data
│   │   ├── expand_multiclass_labels.py ← 8-way augmentation for all 5 classes
│   │   └── download_codebrim_cls.py    ← CODEBRIM classification crop downloader
│   ├── bim/
│   │   └── ifc_writer.py           ← IFCWriter + Pset_DamageInspection
│   ├── pipeline/
│   │   └── pipeline.py             ← end-to-end orchestration
│   └── utils/                      ← shared helpers
├── tests/
│   └── download_data.py            ← Kaggle crack dataset downloader
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

### 5. Rebuild training data from scratch

```bash
# Step 1 — Expand crack labels (500 manual annotations → 4,000 images)
venv\Scripts\python src/detection/expand_manual_labels.py

# Step 2 — Download CODEBRIM classification crops (~8 GB, one-time)
venv\Scripts\python src/detection/download_codebrim_cls.py

# Step 3 — Expand all 5 classes (CODEBRIM + crack → 14,688 images)
venv\Scripts\python src/detection/expand_multiclass_labels.py

# Step 4 — Train
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
| `pytest` | >= 8.0 | Testing |

---

## Dataset Sources

| Class | Source | Images |
|---|---|---|
| crack | [arunrk7/surface-crack-detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (Kaggle) | 40,000 |
| spallation, corrosion, efflorescence, exposed_bars | [CODEBRIM](https://zenodo.org/records/2620293) (Zenodo, CC BY-NC 4.0) | 500 each |

---

## Development Timeline

A chronological record of decisions, dead ends, and wins.

### Phase 1 — Binary crack classifier (mAP: n/a, top-1 accuracy: 99.8%)

Downloaded 40,000 crack/no-crack images from Kaggle. Trained a YOLOv8n-cls classification model. Worked well as a proof of concept but only answered "is there a crack?" — no bounding box, no location.

### Phase 2 — Manual annotation

Sampled 500 images from the crack dataset and labeled them with bounding boxes in Label Studio. This gave us proper YOLO-format detection labels with tight boxes around individual cracks.

**What went wrong:** Label Studio requires a local HTTP server to serve images. Setting up the task JSON with the right URL format took several iterations.

### Phase 3 — Crack detection model (mAP50: 99.4%)

Applied 8-way geometric augmentation to the 500 labels (flip H/V, rotate 90/180/270, transpose, transverse) to get 4,000 training images. Trained YOLOv8n as an object detector — mAP50 99.4% on crack detection.

Augmentation is lossless: all 8 transforms are exact pixel operations with closed-form bounding box math, so no annotation error is introduced.

### Phase 4 — First multi-class attempt (mAP50: 32.3%) — FAILED

Tried to add 4 more damage classes from CODEBRIM's original images using the XML bounding box annotations. The CODEBRIM original dataset has multi-label annotations — most images contain several damage types simultaneously.

**What went wrong:**
- Filtering for single-class images left only 47–202 images per class — far too few.
- The CODEBRIM zip file has a 4 GB offset corruption: all image entries report `header_offset` values that are 4,294,967,296 bytes too large. Reading any image from the zip without correcting this offset returns garbage data ("bad magic number" error). XML metadata files are unaffected.
- mAP50 collapsed to 0.32 — the model barely learned anything.

### Phase 5 — CODEBRIM classification crops (mAP50: 97.8%)

Switched strategy: instead of the original images with sparse multi-label bboxes, used CODEBRIM's *classification dataset* — 500 pre-cropped patches per damage type where the defect fills the entire image.

The trick: treat each crop as a full-image bounding box (`class 0.5 0.5 1.0 1.0`). This gives clean, unambiguous single-class labels with 500 samples per class.

**What went wrong along the way:**

- **PC crash during data preparation:** CODEBRIM classification crops are up to 1,900 × 2,800 px. The original augmentation script accumulated all images in RAM before writing. With 500 × 8 transforms per class, this blew 16 GB of RAM. Fix: rewrite the expander to write each image to disk immediately and resize to 640 × 640 before augmenting.

- **Zero images found for CODEBRIM classes:** The expander used `glob("*.jpg")` but CODEBRIM classification crops are saved as `.png`. Fix: added `glob("*.png")` alongside `.jpg`.

- **ZIP 4 GB offset bug (again):** The same `header_offset` corruption affects the classification zip too. Implemented auto-detection: try reading at `header_offset`; if the PK magic bytes are missing, retry at `header_offset - 4,294,967,296`.

- **Training appeared to crash (PC shutdown):** The first v6-multiclass run completed via early stopping at epoch 75 (patience 30, best at epoch 45). The checkpoint set `start_epoch = 150`, which made the resume logic think training was done — not crashed. The model had already converged.

**Result:** 97.8% mAP50 across all 5 classes, 12,496 train / 2,192 val images.

---

## License

MIT — see [LICENSE](LICENSE) for details.

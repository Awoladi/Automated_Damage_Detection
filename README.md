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

## Model Training History

### Current best: v10 — 57.7% mAP50

| Version | Dataset | mAP@50 | Notes |
|---|---|---|---|
| **v15-rf** | CODEBRIM + Roboflow | *in training* | YOLO11m, Roboflow in val too |
| v14 | Manual + CODEBRIM | ~33% plateau | Peaked epoch 19, slowly declined |
| v13 | Manual + CODEBRIM + Roboflow | ~32% declining | Steadily worse after epoch 25 |
| v12 | 2,041 manual labels | 33.0% | Too few images, overfit |
| **v10** | 13,057 CODEBRIM tiles | **57.7%** | Best deployed model |
| v11 | CODEBRIM + crack tiles | 51.4% | Domain mixing hurt performance |
| v9 | 8,800 CODEBRIM tiles | 57.8% | YOLOv8s |
| v7 | 9,168 CODEBRIM tiles | 57.5% | First tight XML bbox training |
| v4 | 4,000 images | 99.4% | Crack only, single class |

---

## Training Problems & Solutions

### Problem 1 — mAP plateau at 57–58% (v9, v10)

**What happened:** The model kept stalling around 57–58% mAP across multiple runs despite tuning hyperparameters. The dataset was CODEBRIM auto-labeled tiles only (~1,052 unique scenes).

**Root cause:** Limited scene diversity. Only 1,052 unique building photos meant the model saw the same walls over and over even with tiling, causing it to memorise textures rather than learn damage features.

**Solution tried:** Professor recommended YOLO11m + copy_paste augmentation + more epochs + Roboflow datasets.

---

### Problem 2 — Manual labels underperformed (v12, 33%)

**What happened:** 2,041 images were manually annotated in Label Studio across all 5 classes. Training on these alone achieved only 33% mAP.

**Root cause:** Too few unique images. 2,041 images across 5 classes is ~400 per class — not enough for a robust detector. The model overfit quickly.

**Lesson:** Manual labels are high quality but need to supplement a large auto-labeled base, not replace it.

---

### Problem 3 — Roboflow caused steady mAP decline (v13)

**What happened:** Added 5 Roboflow concrete defect datasets (~15,000 images) to training alongside CODEBRIM tiles and manual labels. mAP started at 0.327 at epoch 25 and fell consistently to 0.282 by epoch 63 while training losses kept improving.

**Root cause:** Domain mismatch. Roboflow images come from completely different visual contexts (bridges, floors, industrial settings, varying cameras) while the val set was 100% CODEBRIM building facade close-ups. The model learned to detect damage in Roboflow-style images, but was scored only on CODEBRIM-style ones — so the better it got at Roboflow, the worse its score looked.

**Fix:** Add 20% of Roboflow images to the val set, so the score reflects the full training distribution. When the model improves at Roboflow images, the val mAP goes up, not down.

---

### Problem 4 — 8-way augmentation caused slow decline (v14)

**What happened:** Removed Roboflow, kept Manual + CODEBRIM. Applied 8-way geometric augmentation to manual images (horizontal flip, vertical flip, 90°/180°/270° rotation, transpose, transverse). mAP peaked at 0.330 at epoch 19 then slowly drifted downward.

**Root cause:** The rotated and transposed augmentations (90°, upside-down buildings) created images that don't exist in real facades. The model learned to detect damage in every orientation, but real facade photos always have gravity — so the augmented variety confused the val distribution.

**Fix:** Removed manual labels entirely. They added noise without enough unique scene diversity to justify the complexity.

---

### Current approach — v15-rf

- **Architecture:** YOLO11m (upgrade from YOLOv8m, better small-object detection via STAL)
- **Data:** CODEBRIM auto-labeled tiles + Roboflow 5 datasets (80% train / 20% val split on both)
- **No manual labels** — removed as they introduced noise without sufficient diversity
- **Val set:** ~3,000 Roboflow images + CODEBRIM-derived — reflects actual training distribution
- **Augmentation:** copy_paste=0.3 (pastes rare defect instances into other scenes), standard flips/HSV
- **Epochs:** 200 with patience=50

---

## Damage Classes

| ID | Class | Sources |
|---|---|---|
| 0 | `crack` | CODEBRIM + Roboflow (Concrete 5Nov) |
| 1 | `spallation` | CODEBRIM + Roboflow (Spalling-Det, Concrete 5Nov) |
| 2 | `efflorescence` | CODEBRIM + Roboflow (Efflorescence-Det, Concrete 5Nov) |
| 3 | `exposed_bars` | CODEBRIM + Roboflow (Exposure-Det, Concrete 5Nov) |
| 4 | `corrosion` | CODEBRIM + Roboflow (corrosion x3 severities, Concrete 5Nov) |

---

## Project Structure

```
BIMInspect/
├── app.py                          ← Streamlit dashboard
├── train.py                        ← YOLO11m training script
├── start_labelstudio.bat           ← Launch Label Studio for annotation
├── assets/                         ← logo
├── data/
│   ├── raw/
│   │   ├── CODEBRIM_original_images.zip       ← 7.8 GB (Zenodo)
│   │   ├── CODEBRIM_original_extracted/       ← 1,590 JPGs + 1,052 XMLs
│   │   └── roboflow/                          ← 5 Roboflow dataset ZIPs
│   ├── to_label/                   ← CODEBRIM tiles extracted for Label Studio
│   └── expanded_multiclass/        ← built training dataset (generated)
├── models/
│   ├── weights/                    ← .pt model files (git-ignored)
│   │   ├── best_detection.pt       ← production model (v10, 57.7%)
│   │   └── best.pt                 ← legacy binary classifier (fallback)
│   ├── configs/
│   │   └── dataset_multiclass.yaml ← YOLO dataset config
│   └── exports/                    ← ONNX / TensorRT / CoreML
├── src/
│   ├── detection/
│   │   ├── detector.py                 ← DamageDetector (tiled inference + NMS)
│   │   ├── build_dataset.py            ← Build dataset (CODEBRIM + Roboflow)
│   │   ├── extract_codebrim.py         ← Extract CODEBRIM zip (4 GB offset fix)
│   │   └── extract_for_labeling.py     ← Extract CODEBRIM tiles for Label Studio
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

### 2. Download data
- CODEBRIM: download `CODEBRIM_original_images.zip` from [Zenodo 2620293](https://zenodo.org/records/2620293) → `data/raw/`
- Roboflow: download the 5 dataset ZIPs → `data/raw/roboflow/`

### 3. Extract & build dataset
```bash
# Extract CODEBRIM zip (handles the 4 GB offset bug)
venv\Scripts\python src/detection/extract_codebrim.py

# Build training dataset (CODEBRIM tiles + Roboflow, 80/20 split)
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
| `opencv-python` | Image I/O, tiling, augmentation |
| `streamlit` | Web dashboard |

---

## Dataset Sources

| Source | Images | License |
|---|---|---|
| [CODEBRIM](https://zenodo.org/records/2620293) — building facade damage with XML bbox annotations | 1,590 photos → ~10,000 tiles | CC BY-NC 4.0 |
| [Roboflow: Concrete 5November](https://universe.roboflow.com) — 5 damage classes | 7,443 | CC BY 4.0 |
| [Roboflow: corrosion](https://universe.roboflow.com) — 3 severity levels | 5,473 | CC BY 4.0 |
| [Roboflow: Efflorescence-Det](https://universe.roboflow.com) | 1,011 | CC BY 4.0 |
| [Roboflow: Exposure-Det](https://universe.roboflow.com) | 1,032 | CC BY 4.0 |
| [Roboflow: Spalling-Det](https://universe.roboflow.com) | 1,009 | CC BY 4.0 |

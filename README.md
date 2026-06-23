<p align="center">
  <img src="assets/logo.png" alt="BIMInspect Logo" width="320"/>
</p>

<h1 align="center">BIMInspect</h1>

<p align="center">
  <strong>Automated concrete damage detection for bridge and infrastructure inspection, integrated into BIM.</strong><br/>
  YOLO11 · IFC / ifcopenshell · PyTorch · Streamlit · OpenCV
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-blue?logo=python" />
  <img src="https://img.shields.io/badge/torch-2.10.0%2Bcu126-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/YOLO11-ultralytics-purple" />
  <img src="https://img.shields.io/badge/IFC-ifcopenshell-0078d7" />
  <img src="https://img.shields.io/badge/GPU-RTX%203070-76b900?logo=nvidia" />
  <img src="https://img.shields.io/badge/mAP50-62.3%25-brightgreen" />
</p>

---

## What is BIMInspect?

**BIMInspect** automates concrete damage surveys for bridges and infrastructure. Point a camera at a concrete surface, upload the photo, and every defect is:

1. **Detected** by a fine-tuned YOLO11m model trained primarily on bridge and infrastructure concrete imagery
2. **Localised** with a tight bounding box around the exact defect area
3. **Written into the BIM model** as an `IfcAnnotation` with `Pset_DamageInspection`
4. **Exported** as a ready-to-open IFC file — viewable in Revit, ArchiCAD, or Solibri

The model specialises in concrete surface damage as seen on **bridges, tunnels, and building facades** — it performs best on close-up photos of concrete surfaces similar to its training data.

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

## Damage Classes

| ID | Class | Description |
|---|---|---|
| 0 | `crack` | Hairline to structural cracks in concrete |
| 1 | `spallation` | Concrete surface loss / spalling |
| 2 | `efflorescence` | Salt deposits / moisture ingress |
| 3 | `exposed_bars` | Visible reinforcement bars |
| 4 | `corrosion` | Rebar / steel rust staining |

---

## Model Performance

**Current best: v15-rf — 62.3% mAP50** (+4.6% over previous best)

| Version | mAP@50 | Notes |
|---|---|---|
| **v15-rf** | **62.3%** | YOLO11m · CODEBRIM + Roboflow · Roboflow in val — **production model** |
| v16 | abandoned | Fine-tune from v15-rf, lr too high, destabilised weights |
| v14 | ~33% plateau | CODEBRIM + manual labels, 8-way augmentation caused decline |
| v13 | declining | Roboflow added to train but not val → domain mismatch |
| v12 | 33.0% | Manual labels only, too few unique scenes |
| v10 | 57.7% | Previous best — CODEBRIM only, YOLOv8m |
| v11 | 51.4% | Domain mixing hurt performance |
| v9 | 57.8% | YOLOv8s tiled inference |

---

## Training Problems & Solutions

### Problem 1 — Plateau at 57–58% (v9, v10)
Dataset was CODEBRIM-only (~1,052 unique scenes). Model memorised textures rather than learning damage features. **Fix:** added 5 Roboflow concrete defect datasets (~16k images) for scene diversity.

### Problem 2 — Manual labels underperformed (v12, 33%)
2,041 manually annotated images across 5 classes is ~400 per class — not enough for a robust detector. **Lesson:** manual labels need to supplement a large auto-labeled base, not replace it. Eventually removed entirely.

### Problem 3 — Roboflow caused mAP decline (v13)
Val set was 100% CODEBRIM while training included Roboflow images from completely different visual contexts (bridges, floors, industrial settings). As the model improved on Roboflow images, its CODEBRIM-only val score fell. **Fix:** include 20% of Roboflow in the val set so the score reflects the full training distribution.

### Problem 4 — 8-way augmentation caused slow decline (v14)
Geometric augmentations (90° rotations, transposes) on manual facade images created unnatural orientations. The model learned to detect damage in all orientations, but real photos have gravity — augmented variety confused the val distribution. **Fix:** removed manual labels entirely.

### Problem 5 — Fine-tuning destabilised weights (v16)
Starting a new training run from a trained checkpoint with `lr0=0.01` reset the optimizer at too high a learning rate, causing a 10% mAP drop and never recovering. **Fix:** use `lr0=0.001` when fine-tuning from a pre-trained model.

---

## Project Structure

```
BIMInspect/
├── app.py                          ← Streamlit dashboard
├── train.py                        ← YOLO11m training script
├── start_labelstudio.bat           ← Launch Label Studio for annotation
├── data/
│   ├── raw/
│   │   ├── CODEBRIM_original_images.zip       ← 7.8 GB (Zenodo)
│   │   ├── CODEBRIM_original_extracted/       ← 1,590 JPGs + 1,052 XMLs
│   │   └── roboflow/                          ← 5 Roboflow dataset ZIPs
│   └── expanded_multiclass/        ← built training dataset (generated)
├── models/
│   ├── weights/                    ← .pt model files (git-ignored)
│   │   └── best_detection.pt       ← v15-rf production model (62.3% mAP50)
│   ├── configs/
│   │   └── dataset_multiclass.yaml
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
- **CODEBRIM:** download `CODEBRIM_original_images.zip` from [Zenodo 2620293](https://zenodo.org/records/2620293) → `data/raw/`
- **Roboflow:** download the 5 dataset ZIPs → `data/raw/roboflow/`

### 3. Build dataset
```bash
venv\Scripts\python src/detection/extract_codebrim.py
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

| Source | Images | Domain | License |
|---|---|---|---|
| [CODEBRIM](https://zenodo.org/records/2620293) | ~10,000 tiles | European bridge/building facades | CC BY-NC 4.0 |
| [Roboflow: Concrete 5November](https://universe.roboflow.com) | 7,443 | Mixed concrete infrastructure | CC BY 4.0 |
| [Roboflow: corrosion](https://universe.roboflow.com) | 5,473 | Bridge/industrial corrosion | CC BY 4.0 |
| [Roboflow: Efflorescence-Det](https://universe.roboflow.com) | 1,011 | Concrete walls | CC BY 4.0 |
| [Roboflow: Exposure-Det](https://universe.roboflow.com) | 1,032 | Exposed rebar, bridge decks | CC BY 4.0 |
| [Roboflow: Spalling-Det](https://universe.roboflow.com) | 1,009 | Concrete spalling | CC BY 4.0 |

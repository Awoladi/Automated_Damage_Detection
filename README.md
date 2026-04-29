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

1. **Detected** by a fine-tuned YOLOv8s object detection model across 5 damage classes
2. **Localised** with a tight bounding box around the exact defect area
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
│  1. YOLOv8s         │  src/detection/detector.py
│  Object Detection   │  • best_detection.pt  (5 damage classes)
│                     │  • Tight bboxes per defect
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

| Version | Epochs | mAP@50 | Notes |
|---|---|---|---|
| **v9 — tiled dataset** | 121 (best ep. 96) | **57.8%** | 640×640 tiles from original-res images, class-balanced aug |
| v8 — oversampled | 83 (best ep. 44) | 57.0% | YOLOv8s, rare-class oversampling, resize to 640 |
| v7 — real bboxes | 75 | 57.5% | YOLOv8n, tight XML bboxes, resize to 640 |
| v6 — full-image labels | 75 | 97.8%* | *bbox covered whole image — not usable |
| v4 — crack only | 150 | 99.4% | Manual annotations, single class |

`best_detection.pt` currently holds v9 weights.

---

## Damage Classes

| ID | Class | Training Source |
|---|---|---|
| 0 | `crack` | 500 hand-labeled images × 8-way augmentation = 4,000 images |
| 1 | `spallation` | CODEBRIM original images — 640×640 tiles from full resolution |
| 2 | `efflorescence` | CODEBRIM original images — 640×640 tiles from full resolution |
| 3 | `exposed_bars` | CODEBRIM original images — 640×640 tiles from full resolution |
| 4 | `corrosion` | CODEBRIM original images — 640×640 tiles from full resolution |

Dataset (v9): ~8,800 train / 1,600 val. Class-balanced augmentation ensures equal representation across all 5 classes.

---

## Project Structure

```
BIMInspect/
├── app.py                          ← Streamlit dashboard
├── train.py                        ← YOLOv8 training script
├── assets/                         ← logo and static resources
├── data/
│   ├── raw/
│   │   ├── Positive/               ← 20,000 crack images (Kaggle)
│   │   ├── Negative/               ← 20,000 no-crack images (Kaggle)
│   │   ├── CODEBRIM_original_images.zip        ← 7.8 GB, bbox annotations
│   │   └── CODEBRIM_classification_dataset.zip ← 7.4 GB (no longer used)
│   ├── annotated/                  ← 500 manually labeled crack images (YOLO format)
│   ├── expanded_manual/            ← 4,000 augmented crack training images
│   └── expanded_multiclass/        ← full 5-class training dataset (generated)
├── models/
│   ├── weights/                    ← .pt model files (git-ignored)
│   │   ├── best_detection.pt       ← current production model
│   │   └── best.pt                 ← legacy binary crack classifier (fallback)
│   ├── configs/
│   │   └── dataset_multiclass.yaml ← training dataset config
│   └── exports/                    ← ONNX / TensorRT / CoreML
├── src/
│   ├── detection/
│   │   ├── detector.py                 ← DamageDetector inference class
│   │   ├── expand_manual_labels.py     ← 8-way augmentation for crack data
│   │   └── expand_codebrim_bbox.py     ← CODEBRIM tiled dataset builder
│   ├── bim/
│   │   └── ifc_writer.py               ← IFCWriter + Pset_DamageInspection
│   ├── pipeline/
│   │   └── pipeline.py                 ← end-to-end orchestration
│   └── utils/                          ← shared helpers
├── tests/
│   └── download_data.py                ← Kaggle crack dataset downloader
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Download datasets

**Crack data** (Kaggle):
```bash
venv\Scripts\python tests/download_data.py
```

**CODEBRIM original images** (~7.8 GB, from [Zenodo record 2620293](https://zenodo.org/records/2620293)):
Download `CODEBRIM_original_images.zip` manually and place it at `data/raw/CODEBRIM_original_images.zip`.

### 4. Build training dataset
```bash
# Step 1 — Expand crack labels (500 manual annotations → 4,000 images)
venv\Scripts\python src/detection/expand_manual_labels.py

# Step 2 — Build full 5-class dataset: tile CODEBRIM at full resolution + balance classes
venv\Scripts\python src/detection/expand_codebrim_bbox.py
```

### 5. Train
```bash
venv\Scripts\python train.py
```

### 6. Launch dashboard
```bash
venv\Scripts\python -m streamlit run app.py
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

| Class | Source | License |
|---|---|---|
| crack / no-crack | [arunrk7/surface-crack-detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (Kaggle) | Public |
| spallation, corrosion, efflorescence, exposed_bars | [CODEBRIM](https://zenodo.org/records/2620293) (Zenodo) | CC BY-NC 4.0 |

---

## Development Timeline

A full record of how the project was built — decisions made, approaches tried, what failed and why.

---

### Phase 1 — Project setup and binary crack classifier

**Goal:** Prove that a camera-based AI can detect structural cracks at all, before committing to a full BIM pipeline.

Downloaded 40,000 crack / no-crack surface images from the Kaggle dataset `arunrk7/surface-crack-detection` (20,000 positive, 20,000 negative, 227×227 px JPEGs). Trained a YOLOv8n-cls classification model for 20 epochs.

**Result:** 99.8% top-1 accuracy — the model could reliably distinguish cracked from uncracked surfaces.

**Limitation:** A classifier answers "is there a crack?" but gives no bounding box, no location. Not useful for BIM annotation, which needs precise pixel coordinates to map damage back to 3-D building geometry.

---

### Phase 2 — Manual bounding box annotation

**Goal:** Get proper YOLO-format detection labels with tight bounding boxes around individual cracks.

Sampled 500 images from the crack dataset, set up Label Studio with a local HTTP file server, and manually drew bounding boxes on each image. This took multiple sessions to complete.

**Problems encountered:**
- Label Studio requires images to be served over HTTP, not loaded from the local filesystem. The task JSON had to be regenerated several times to get the URL format right.
- The Label Studio export format needed conversion to YOLO `.txt` format (one file per image, normalised `cx cy w h` coordinates).

**Result:** 500 images with hand-drawn, tight bounding boxes in YOLO format stored in `data/annotated/`.

---

### Phase 3 — 8-way geometric augmentation and crack detection model

**Goal:** Expand 500 labeled images to a training set large enough for reliable detection, without introducing label errors.

Applied 8 exact geometric transforms to every labeled image: original, horizontal flip, vertical flip, rotate 90°/180°/270°, transpose (flip along main diagonal), transverse (flip along anti-diagonal). All 8 transforms have closed-form bounding-box math — no interpolation, no coordinate rounding. 500 × 8 = 4,000 training images.

Trained YOLOv8n as an object detector (not a classifier) for 150 epochs on the 4,000 images.

**Result:** mAP@50 = **99.4%**, Precision = 99.1%, Recall = 99.2%. Clean bounding boxes around individual cracks.

Built the full BIM pipeline around this model: `detector.py` → `pipeline.py` → `ifc_writer.py` → Streamlit dashboard.

---

### Phase 4 — First multi-class attempt using CODEBRIM bounding box annotations (FAILED — mAP 32.3%)

**Goal:** Add 4 more damage classes (spallation, efflorescence, exposed bars, corrosion) using the CODEBRIM dataset's original images and their Pascal VOC bounding box annotations.

CODEBRIM provides 1,590 full building photos with per-image XML files containing annotated bounding boxes and multi-label damage type flags.

**Problems encountered:**

- **ZIP 4 GB offset corruption.** The CODEBRIM original images zip (`CODEBRIM_original_images.zip`, 7.8 GB) has a structural defect: every image entry's `header_offset` field is inflated by exactly 4,294,967,296 bytes (2³²). Reading any image at the reported offset returns garbage. The XML annotation files are unaffected because they sit near the start of the archive. Fix: when the PK magic bytes (`\x50\x4B\x03\x04`) are not found at `header_offset`, retry at `header_offset − 4,294,967,296`.

- **Too few single-class images.** To avoid ambiguous labels, only images where exactly one damage type was active were selected. This left 47–202 images per class — far too few for reliable detection.

**Result:** mAP@50 collapsed to **0.323** — the model barely learned anything. Per-class AP was near zero for all non-crack classes.

---

### Phase 5 — Classification crop workaround (mAP 97.8%, but imprecise bboxes)

**Goal:** Fix the data shortage by using CODEBRIM's classification dataset instead — pre-cropped 500-image patches per damage type where the defect fills the whole crop.

**Approach:** Label each crop as a full-image bounding box: `class_id 0.5 0.5 1.0 1.0`. This gives 500 clean, unambiguous single-class samples per damage type. Apply 8-way augmentation → ~4,000 images per class → ~16,000 total.

**Problems encountered:**

- **PC crash during augmentation.** CODEBRIM classification crops are up to 1,900×2,800 px. The original augmentation script accumulated all images in RAM before writing to disk. With 500 × 8 transforms per class across 4 classes simultaneously, this exceeded 16 GB of RAM. Fix: rewrite the expander to process one image at a time, resize to 640×640 before augmenting, and write to disk immediately.

- **Zero images found.** The augmentation script used `glob("*.jpg")` but CODEBRIM classification crops are saved as `.png`. Fix: also glob for `*.png`.

- **ZIP 4 GB offset (again).** The classification dataset zip has the same offset corruption. Same fix applied.

- **Training appeared to crash.** The first run completed via early stopping at epoch 75 (patience 30, best at epoch 45). The checkpoint set `start_epoch = 150`, which made the resume logic report "nothing to resume." The model had already converged — it was not a crash.

**Result:** mAP@50 = **97.8%** across all 5 classes. The model detected damage types correctly.

**New problem discovered:** Because every training label was `class_id 0.5 0.5 1.0 1.0`, the model learned to predict near-full-image bounding boxes for all non-crack classes. Uploading a building photo to the dashboard correctly identified corrosion at 96.95% confidence but drew a bounding box over the entire building facade rather than the affected area.

---

### Phase 6 — Proper CODEBRIM bbox training (v7 — mAP 57.5%)

**Goal:** Fix the bounding box precision problem by retraining with actual annotated tight bboxes from the CODEBRIM original images.

**Approach:** Return to `CODEBRIM_original_images.zip` with the offset fix already in place, but this time use ALL annotated images (not just single-class ones). Each `<object>` in the per-image XML files contains a `<bndbox>` with `xmin/ymin/xmax/ymax` pixel coordinates and a multi-label `<Defect>` block. For every active damage type in an object, emit one YOLO annotation at those exact coordinates. One image can produce multiple annotations across different classes — this is valid YOLO format. Applied 8-way geometric augmentation and resized all images to 640×640 before training.

Dataset stats: 1,052 annotated CODEBRIM images → ×8 augmented = ~6,792 tiles + 3,400 crack images = **9,168 train / 1,624 val**.

Training: 75 epochs, patience 20, YOLOv8n, batch 32, 640×640, GPU RTX 3070.

**Result:** mAP@50 = **57.5%**. Bounding boxes are now tight around actual defects. The mAP is limited because resizing 4,608×3,456 building photos to 640×640 shrinks defects to 10–30 px — too small for reliable detection.

---

### Phase 7 — Tiling at full resolution (v8/v9 — mAP 57.8%)

**Goal:** Improve detection of small defects by preserving their pixel size during training.

**Diagnosis:** CODEBRIM original photos are ~4,608×3,456 px. A defect annotation covering 300×200 px in the original becomes a 40×27 px blob after resizing to 640 — far below the typical 32 px minimum for reliable YOLO detection. The model was never seeing defects at a useful scale.

**Approach:** Instead of resizing whole images, cut 640×640 tiles from the original-resolution images at a stride of 480 px (25% overlap). A tile is only kept if it contains at least one bounding box with ≥25% visible area. This means each defect is seen at approximately its original pixel size, while the tile fits directly into YOLO's input without any downscaling.

Also tested class oversampling (v8, YOLOv8s): rare classes (efflorescence, exposed_bars) were duplicated with extra augmentation passes. No meaningful improvement over v7.

Dataset stats (v9, tiled): ~5,400 CODEBRIM tiles + 3,400 crack images = **8,800 train / 1,600 val**.

Training: 150 epochs (early-stopped at 121, best at epoch 96), YOLOv8s, batch 16, GPU RTX 3070.

**Result:** mAP@50 = **57.8%**. Modest gain from tiling alone. Root cause of the plateau: severe class imbalance — crack has 3,400 images while efflorescence and exposed_bars have fewer than 600 tiles each.

---

### Phase 8 — Class-balanced augmentation (current)

**Goal:** Eliminate the class imbalance that is limiting non-crack detection.

**Approach:** After generating all tiles, count how many training images contain each class. Identify the class with the most images (the target). For every underrepresented class, randomly sample existing images of that class and apply one of the 7 geometric augmentation transforms until all classes reach the target count. This guarantees equal representation without discarding any data.

`expand_codebrim_bbox.py` now performs: (1) full-resolution tiling, (2) crack data copy, (3) per-class balancing via augmentation.

Training run in progress.

---

## License

MIT — see [LICENSE](LICENSE) for details.

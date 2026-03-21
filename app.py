"""
BIMInspect — Streamlit Dashboard
Upload a photo, run damage detection, download the annotated IFC file.
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Make src importable when running from project root ─────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.detector import DamageDetector, DEFAULT_WEIGHTS
from src.pipeline.pipeline import run

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "BIMInspect",
    page_icon  = "assets/logo.png",
    layout     = "wide",
)

# ── Load model once (cached across reruns) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_detector() -> DamageDetector:
    return DamageDetector(weights=DEFAULT_WEIGHTS)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("---")
    st.subheader("Settings")
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01,
        help="Detections below this value are treated as no damage."
    )
    storey_name = st.text_input("IFC storey name", value="Ground Floor")
    st.markdown("---")
    st.caption("BIMInspect v1.0 · YOLOv8n-cls · IFC4")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("BIMInspect")
st.markdown(
    "Upload a construction photo to detect structural damage and export "
    "the result as an annotated **IFC BIM file**."
)
st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a photo (JPG / PNG / BMP)",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)

if not uploaded:
    st.info("Upload an image above to start the inspection.")
    st.stop()

# ── Decode image for display ───────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w       = img_bgr.shape[:2]

# ── Run pipeline ───────────────────────────────────────────────────────────────
detector = load_detector()

with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
    tmp.write(file_bytes.tobytes())
    tmp_path = Path(tmp.name)

with st.spinner("Running damage detection…"):
    result = run(
        image_path           = tmp_path,
        detector             = detector,
        storey_name          = storey_name,
        skip_no_damage       = False,   # always write so the IFC is downloadable
        confidence_threshold = confidence_threshold,
    )
tmp_path.unlink(missing_ok=True)

# ── Layout: two columns ────────────────────────────────────────────────────────
col_img, col_info = st.columns([3, 2], gap="large")

# ── Left — annotated image ─────────────────────────────────────────────────────
with col_img:
    st.subheader("Inspection photo")

    annotated = img_rgb.copy()

    if result.bbox and result.damage_class == "crack":
        x1, y1, x2, y2 = result.bbox
        # Draw bounding box (red in RGB)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 38, 38), 2)
        # Label above box
        label     = f"crack {result.confidence * 100:.1f}%"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(w, h) / 400)
        thickness  = max(1, int(font_scale * 2))
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tag_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 4, y1), (220, 38, 38), -1)
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - baseline - 2),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    st.image(annotated, use_container_width=True)

# ── Right — results ────────────────────────────────────────────────────────────
with col_info:
    st.subheader("Detection result")

    is_crack = (result.damage_class == "crack"
                and result.confidence >= confidence_threshold)

    if is_crack:
        st.error("Damage detected", icon="🚨")
    else:
        st.success("No damage detected", icon="✅")

    # Metrics row
    m1, m2 = st.columns(2)
    m1.metric("Class",      result.damage_class.replace("_", " ").title())
    m2.metric("Confidence", f"{result.confidence * 100:.2f}%")

    # Bounding box details
    if result.bbox:
        st.markdown("**Bounding box (pixels)**")
        x1, y1, x2, y2 = result.bbox
        nx1, ny1, nx2, ny2 = result.bbox_normalized or (0, 0, 0, 0)
        bbox_cols = st.columns(4)
        for col, label, val in zip(
            bbox_cols,
            ["x1", "y1", "x2", "y2"],
            [x1, y1, x2, y2],
        ):
            col.metric(label, val)

        st.markdown("**Bounding box (normalised)**")
        st.code(
            f"x1={nx1:.4f}  y1={ny1:.4f}\n"
            f"x2={nx2:.4f}  y2={ny2:.4f}",
            language=None,
        )
    else:
        st.info("No bounding box (no crack localised).")

    st.markdown(f"**Image size:** {w} × {h} px")

    # ── IFC download ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("IFC export")

    if result.ifc_output_path and result.ifc_output_path.exists():
        ifc_bytes = result.ifc_output_path.read_bytes()
        st.download_button(
            label     = "Download IFC file",
            data      = ifc_bytes,
            file_name = result.ifc_output_path.name,
            mime      = "application/x-step",
            use_container_width = True,
            type      = "primary",
        )
        st.caption(
            f"`{result.ifc_output_path.name}` — "
            f"IfcAnnotation with Pset\\_DamageInspection"
        )
    else:
        st.warning("IFC file could not be generated.")

    if result.error:
        st.error(f"Pipeline error: {result.error}")

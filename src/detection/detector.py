"""
BIMInspect — Damage Detector
Supports two model modes, chosen automatically based on which weights are found:

  1. Detection mode  (best_detection.pt — YOLOv8n)
     Preferred. Returns bounding boxes directly from the network.
     Fast, accurate, no gradient computation needed.

  2. Classification mode  (best.pt — YOLOv8n-cls)
     Fallback. Runs the classifier then estimates a bbox via Grad-CAM on the
     last convolutional layer of the classification head.

DEFAULT_WEIGHTS resolves to best_detection.pt if it exists, else best.pt.

Returns DetectionResult with:
    damage_class    : "crack" | "no_crack"
    confidence      : 0.0 – 1.0
    bbox            : (x1, y1, x2, y2) pixel coords  — None when no crack
    bbox_normalized : (x1, y1, x2, y2) in 0-1 range — None when no crack
    model_mode      : "detection" | "classification"
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT              = Path(__file__).resolve().parents[2]
_WEIGHTS_DIR      = ROOT / "models" / "weights"
_DET_WEIGHTS      = _WEIGHTS_DIR / "best_detection.pt"
_CLS_WEIGHTS      = _WEIGHTS_DIR / "best.pt"
DEFAULT_WEIGHTS   = _DET_WEIGHTS if _DET_WEIGHTS.exists() else _CLS_WEIGHTS

CRACK_CLASS       = "crack"
HEATMAP_THRESHOLD = 0.4     # Grad-CAM fallback threshold


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    image_path:       str
    damage_class:     str                                        # "crack" | "no_crack"
    confidence:       float                                      # 0.0 – 1.0
    bbox:             tuple[int, int, int, int] | None = None   # pixels (x1,y1,x2,y2)
    bbox_normalized:  tuple[float, float, float, float] | None = None
    img_width:        int = 0
    img_height:       int = 0
    class_id:         int = 0
    model_mode:       str = "detection"                          # "detection" | "classification"

    def __str__(self) -> str:
        lines = [
            f"Image      : {self.image_path}",
            f"Class      : {self.damage_class}  (id={self.class_id})",
            f"Confidence : {self.confidence:.4f}",
            f"Mode       : {self.model_mode}",
        ]
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            lines.append(f"BBox (px)  : x1={x1} y1={y1} x2={x2} y2={y2}")
            lines.append(f"BBox (norm): {tuple(f'{v:.4f}' for v in self.bbox_normalized)}")
        else:
            lines.append("BBox       : None  (no damage detected)")
        return "\n".join(lines)


# ── Grad-CAM (classification fallback) ────────────────────────────────────────

class _GradCAM:
    """Hooks into model.9.conv.conv (last conv of YOLOv8n-cls head)."""

    def __init__(self, inner: torch.nn.Module) -> None:
        self._acts:  torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        layer = inner.model[9].conv.conv
        self._fh = layer.register_forward_hook(
            lambda _m, _i, o: setattr(self, '_acts', o.detach())
        )
        self._bh = layer.register_full_backward_hook(
            lambda _m, _gi, go: setattr(self, '_grads', go[0].detach())
        )

    def compute(self, logits: torch.Tensor, cls: int, hw: tuple[int, int]) -> np.ndarray:
        logits[0, cls].backward(retain_graph=True)
        w   = self._grads[0].mean(dim=(1, 2))
        cam = F.relu(torch.einsum("c,chw->hw", w, self._acts[0]))
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
        return (cam / cam.max()).astype(np.float32) if cam.max() > 0 else cam

    def remove(self) -> None:
        self._fh.remove()
        self._bh.remove()


def _bbox_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = HEATMAP_THRESHOLD,
) -> tuple[int, int, int, int] | None:
    binary   = (heatmap >= threshold).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return None
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    x1 = int(stats[best, cv2.CC_STAT_LEFT])
    y1 = int(stats[best, cv2.CC_STAT_TOP])
    x2 = x1 + int(stats[best, cv2.CC_STAT_WIDTH])
    y2 = y1 + int(stats[best, cv2.CC_STAT_HEIGHT])
    return (x1, y1, x2, y2)


# ── Main detector class ────────────────────────────────────────────────────────

class DamageDetector:
    """
    Loads a YOLOv8 model and detects structural damage in images.
    Automatically uses detection mode (native bboxes) or classification
    mode (Grad-CAM bboxes) based on the loaded weights.

    Usage:
        detector = DamageDetector()
        result   = detector.detect("path/to/image.jpg")
        print(result)
    """

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights}\n"
                "Run src/detection/train.py (classifier) or "
                "src/detection/train_detection.py (detector) first."
            )
        self.model      = YOLO(str(weights))
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.names      = self.model.names
        self.model_mode = self._infer_mode()
        self._crack_idx = next(
            (k for k, v in self.names.items() if v == CRACK_CLASS), 0
        )
        print(f"DamageDetector loaded [{self.model_mode}]: {weights.name}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, image_path: str | Path) -> DetectionResult:
        """
        Run inference on a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            DetectionResult with class, confidence, and optional bbox.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = img_bgr.shape[:2]

        if self.model_mode == "detection":
            return self._detect_detection_model(img_bgr, image_path, h, w)
        else:
            return self._detect_classification_model(img_bgr, image_path, h, w)

    # ── Detection-model path (native bboxes) ───────────────────────────────────

    def _detect_detection_model(
        self, img_bgr: np.ndarray, image_path: Path, h: int, w: int
    ) -> DetectionResult:
        results = self.model.predict(img_bgr, verbose=False)
        boxes   = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                image_path   = str(image_path),
                damage_class = "no_crack",
                confidence   = 0.0,
                img_width    = w,
                img_height   = h,
                model_mode   = "detection",
            )

        # Pick the highest-confidence crack box
        best_idx  = int(boxes.conf.argmax())
        conf      = float(boxes.conf[best_idx])
        cls_id    = int(boxes.cls[best_idx])
        class_name = self.names.get(cls_id, "crack")

        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        bbox      = (x1, y1, x2, y2)
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)

        return DetectionResult(
            image_path      = str(image_path),
            damage_class    = class_name,
            confidence      = conf,
            bbox            = bbox,
            bbox_normalized = bbox_norm,
            img_width       = w,
            img_height      = h,
            class_id        = cls_id,
            model_mode      = "detection",
        )

    # ── Classification-model path (Grad-CAM bboxes) ────────────────────────────

    def _detect_classification_model(
        self, img_bgr: np.ndarray, image_path: Path, h: int, w: int
    ) -> DetectionResult:
        results    = self.model.predict(img_bgr, verbose=False)
        probs      = results[0].probs
        class_id   = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = self.names[class_id]

        bbox = bbox_normalized = None
        if class_name == CRACK_CLASS:
            bbox, bbox_normalized = self._gradcam_bbox(img_bgr, class_id, h, w)

        return DetectionResult(
            image_path      = str(image_path),
            damage_class    = class_name,
            confidence      = confidence,
            bbox            = bbox,
            bbox_normalized = bbox_normalized,
            img_width       = w,
            img_height      = h,
            class_id        = class_id,
            model_mode      = "classification",
        )

    def _gradcam_bbox(
        self, img_bgr: np.ndarray, class_id: int, h: int, w: int
    ) -> tuple[tuple[int, int, int, int] | None,
               tuple[float, float, float, float] | None]:
        inner = self.model.model
        inner.to(self.device)
        inner.train()

        saved = []
        for m in inner.modules():
            if hasattr(m, "inplace"):
                saved.append((m, m.inplace))
                m.inplace = False

        gc = _GradCAM(inner)
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t = (
                torch.tensor(cv2.resize(img_rgb, (224, 224)), dtype=torch.float32)
                .permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            )
            t.requires_grad_(True)
            out    = inner(t)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            heatmap = gc.compute(logits, class_id, hw=(h, w))
            bbox_px = _bbox_from_heatmap(heatmap)
        finally:
            gc.remove()
            inner.eval()
            for m, s in saved:
                m.inplace = s

        if bbox_px is None:
            return None, None

        x1, y1, x2, y2 = bbox_px
        return bbox_px, (x1 / w, y1 / h, x2 / w, y2 / h)

    # ── Helper ─────────────────────────────────────────────────────────────────

    def _infer_mode(self) -> str:
        """Determine whether the loaded model is a detector or classifier."""
        task = getattr(self.model, "task", None)
        if task == "detect":
            return "detection"
        if task == "classify":
            return "classification"
        # Fallback: inspect the final model layer name
        try:
            last = list(self.model.model.model.children())[-1]
            return "classification" if "Classify" in type(last).__name__ else "detection"
        except Exception:
            return "detection"


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path> [weights_path]")
        sys.exit(1)

    image   = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_WEIGHTS

    detector = DamageDetector(weights=weights)
    result   = detector.detect(image)
    print(result)

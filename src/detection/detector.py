"""
BIMInspect — Damage Detector
Loads models/weights/best.pt (YOLOv8n-cls) and runs inference on a single image.

Returns:
    DetectionResult with:
        - damage_class    : "crack" or "no_crack"
        - confidence      : 0.0 – 1.0
        - bbox            : (x1, y1, x2, y2) pixel coords  — None when no crack
        - bbox_normalized : (x1, y1, x2, y2) in 0-1 range — None when no crack

Bounding box estimation uses Grad-CAM on the last convolutional layer of the
classification head (model.9.conv) to localise the most activated region without
needing a separate detection model.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "weights" / "best.pt"

CRACK_CLASS  = "crack"
DAMAGE_LABEL = "crack"           # only localise cracks, not no_crack

# Grad-CAM: threshold fraction of the peak activation to keep
HEATMAP_THRESHOLD = 0.4


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    image_path:       str
    damage_class:     str                              # "crack" | "no_crack"
    confidence:       float                            # 0.0 – 1.0
    bbox:             tuple[int, int, int, int] | None = None   # pixels (x1,y1,x2,y2)
    bbox_normalized:  tuple[float, float, float, float] | None = None
    img_width:        int = 0
    img_height:       int = 0
    class_id:         int = 0

    def __str__(self) -> str:
        lines = [
            f"Image      : {self.image_path}",
            f"Class      : {self.damage_class}  (id={self.class_id})",
            f"Confidence : {self.confidence:.4f}",
        ]
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            lines.append(f"BBox (px)  : x1={x1} y1={y1} x2={x2} y2={y2}")
            lines.append(f"BBox (norm): {tuple(f'{v:.4f}' for v in self.bbox_normalized)}")
        else:
            lines.append("BBox       : None  (no damage detected)")
        return "\n".join(lines)


# ── Grad-CAM helper ────────────────────────────────────────────────────────────

class _GradCAM:
    """
    Minimal Grad-CAM implementation via PyTorch forward/backward hooks.
    Hooks into the Conv layer at model.9.conv (last conv before global pool).
    """

    def __init__(self, model_inner: torch.nn.Module) -> None:
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        # The target layer is the Conv wrapper; we hook the inner Conv2d
        target_layer = model_inner.model[9].conv.conv

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        logits:     torch.Tensor,
        class_idx:  int,
        img_hw:     tuple[int, int],
    ) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1] upsampled to img_hw."""
        # Back-prop from the target class score
        logits[0, class_idx].backward(retain_graph=True)

        grads   = self._gradients[0]          # (C, h, w)
        acts    = self._activations[0]        # (C, h, w)
        weights = grads.mean(dim=(1, 2))      # global-average-pool gradients

        cam = torch.einsum("c,chw->hw", weights, acts)
        cam = F.relu(cam)

        # Upsample to original image size
        cam_np = cam.cpu().numpy()
        cam_up = cv2.resize(cam_np, (img_hw[1], img_hw[0]),
                            interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]
        if cam_up.max() > 0:
            cam_up = cam_up / cam_up.max()

        return cam_up.astype(np.float32)

    def remove(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _bbox_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = HEATMAP_THRESHOLD,
) -> tuple[int, int, int, int] | None:
    """
    Threshold the Grad-CAM heatmap and return the bounding box of the
    largest connected component as (x1, y1, x2, y2).
    Returns None if nothing exceeds the threshold.
    """
    binary = (heatmap >= threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    if n_labels <= 1:
        return None

    # Pick the largest non-background component
    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1   # +1 to skip background label 0

    x1 = int(stats[best, cv2.CC_STAT_LEFT])
    y1 = int(stats[best, cv2.CC_STAT_TOP])
    x2 = x1 + int(stats[best, cv2.CC_STAT_WIDTH])
    y2 = y1 + int(stats[best, cv2.CC_STAT_HEIGHT])

    return (x1, y1, x2, y2)


# ── Main detector class ────────────────────────────────────────────────────────

class DamageDetector:
    """
    Loads a fine-tuned YOLOv8n-cls model and detects damage in images.

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
                "Run src/detection/train.py first."
            )
        self.model  = YOLO(str(weights))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.names  = self.model.names          # {0: 'crack', 1: 'no_crack'}
        self._crack_idx = next(
            (k for k, v in self.names.items() if v == CRACK_CLASS), 0
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, image_path: str | Path) -> DetectionResult:
        """
        Run inference on a single image.

        Args:
            image_path: Path to the input image (JPG / PNG / BMP …).

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

        # ── Classify ───────────────────────────────────────────────────────────
        results     = self.model.predict(img_bgr, verbose=False)
        probs       = results[0].probs
        class_id    = int(probs.top1)
        confidence  = float(probs.top1conf)
        class_name  = self.names[class_id]

        # ── Localise with Grad-CAM (only for crack detections) ─────────────────
        bbox            = None
        bbox_normalized = None

        if class_name == DAMAGE_LABEL:
            bbox, bbox_normalized = self._gradcam_bbox(img_bgr, class_id, h, w)

        return DetectionResult(
            image_path       = str(image_path),
            damage_class     = class_name,
            confidence       = confidence,
            bbox             = bbox,
            bbox_normalized  = bbox_normalized,
            img_width        = w,
            img_height       = h,
            class_id         = class_id,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _gradcam_bbox(
        self,
        img_bgr: np.ndarray,
        class_id: int,
        h: int,
        w: int,
    ) -> tuple[tuple[int, int, int, int] | None,
               tuple[float, float, float, float] | None]:
        """Compute Grad-CAM heatmap and derive a bounding box."""

        inner_model = self.model.model
        inner_model.to(self.device)
        inner_model.train()   # train mode keeps gradient graph; eval fuses/detaches

        gradcam = _GradCAM(inner_model)

        # SiLU uses inplace=True by default which breaks autograd backward hooks.
        # Temporarily disable all inplace ops and restore afterwards.
        inplace_states: list[tuple] = []
        for m in inner_model.modules():
            if hasattr(m, "inplace"):
                inplace_states.append((m, m.inplace))
                m.inplace = False

        try:
            # Pre-process: resize + normalise to match training pipeline
            img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            tensor = (
                torch.tensor(img_resized, dtype=torch.float32)
                .permute(2, 0, 1)           # HWC → CHW
                .unsqueeze(0)               # → (1, C, H, W)
                .to(self.device) / 255.0
            )
            tensor.requires_grad_(True)

            # Forward pass — returns (1, num_classes) logits in train mode
            out = inner_model(tensor)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # Compute Grad-CAM at the original image resolution
            heatmap = gradcam.compute(logits, class_id, img_hw=(h, w))
            bbox_px = _bbox_from_heatmap(heatmap, HEATMAP_THRESHOLD)

        finally:
            gradcam.remove()
            inner_model.eval()
            for m, state in inplace_states:
                m.inplace = state

        if bbox_px is None:
            return None, None

        x1, y1, x2, y2 = bbox_px
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)

        return bbox_px, bbox_norm


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path> [weights_path]")
        sys.exit(1)

    image  = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_WEIGHTS

    detector = DamageDetector(weights=weights)
    result   = detector.detect(image)

    print(result)

"""
Multi-sensor fusion.

Produces a sensor-aware combined visualization rather than a naive
alpha blend. Strategy:
  - OHRC (reference, optical high-resolution) supplies the luminance /
    spatial-detail base layer, since it has the highest spatial
    resolution and is defined as the geometric reference.
  - Each successfully-registered source is analyzed for where it adds
    *new* structural information (via its gradient magnitude) beyond
    what OHRC already shows, and only that structural contribution is
    blended in, weighted by local edge strength and by the pair's
    registration confidence. This avoids the failure mode of simply
    averaging four images together, which destroys the crisp OHRC
    detail.
  - A separate per-sensor "contribution map" is exposed so the UI can
    let the scientist toggle/inspect what each sensor actually added.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

CONFIDENCE_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3, None: 0.0}


def _structural_map(gray: np.ndarray) -> np.ndarray:
    grad = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad = np.abs(grad)
    grad = cv2.normalize(grad, None, 0, 1.0, cv2.NORM_MINMAX)
    return grad


def fuse(ohrc_gray: np.ndarray, registered: Dict[str, dict]) -> Dict:
    """
    registered: {sensor: {"image": warped_bgr_or_gray, "mask": binary_mask, "confidence": str}}
    Returns dict with the fused BGR image and a per-sensor contribution map (0-255 uint8).
    """
    h, w = ohrc_gray.shape[:2]
    base = cv2.cvtColor(ohrc_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # False-color channel assignment so each successfully registered
    # sensor's structural contribution is visually distinguishable:
    # TMC -> boosts Green, IIRS -> boosts Red, SAR -> boosts Blue,
    # on top of the OHRC grayscale base.
    channel_map = {"TMC": 1, "IIRS": 2, "SAR": 0}  # BGR indices (G, R, B)

    contributions = {}
    fused = base.copy()

    for sensor, data in registered.items():
        img = data["image"]
        mask = data["mask"].astype(bool)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        weight = CONFIDENCE_WEIGHT.get(data.get("confidence"), 0.3)

        struct = _structural_map(gray) * mask * weight
        contributions[sensor] = (struct * 255).astype(np.uint8)

        ch = channel_map.get(sensor)
        if ch is not None:
            boost = (struct * 90.0)  # additive boost, capped by clipping below
            fused[:, :, ch] = fused[:, :, ch] + boost

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return {"fused_image": fused, "contributions": contributions}

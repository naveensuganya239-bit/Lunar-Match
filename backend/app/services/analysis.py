"""
Image characteristics analysis.

Computes real, measured statistics from each image (intensity
distribution, contrast, edge/texture density) so that downstream
preprocessing and registration decisions are informed by the actual
pixel data rather than assumptions.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import ImageCharacteristics


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def analyze_image(sensor: str, img: np.ndarray) -> ImageCharacteristics:
    gray = to_gray(img)
    h, w = gray.shape
    channels = 1 if img.ndim == 2 else img.shape[2]

    mean_i = float(np.mean(gray))
    std_i = float(np.std(gray))

    lo, hi = np.percentile(gray, [1, 99])
    michelson = float((hi - lo) / (hi + lo + 1e-6))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(h * w)

    dynamic_range = int(gray.max()) - int(gray.min())

    notes = []
    if std_i < 15:
        notes.append("Low intensity variance -- image may be low-contrast; CLAHE recommended.")
    if edge_density < 0.01:
        notes.append("Very low edge density -- may yield sparse keypoints; multi-scale detection recommended.")
    if dynamic_range < 60:
        notes.append("Narrow dynamic range detected.")

    if edge_density > 0.06:
        feature_density = "HIGH"
    elif edge_density > 0.02:
        feature_density = "MEDIUM"
    else:
        feature_density = "LOW"

    return ImageCharacteristics(
        sensor=sensor,
        width=w,
        height=h,
        channels=channels,
        mean_intensity=round(mean_i, 2),
        std_intensity=round(std_i, 2),
        contrast_michelson=round(michelson, 4),
        edge_density=round(edge_density, 5),
        estimated_feature_density=feature_density,
        dynamic_range=dynamic_range,
        notes=notes,
    )

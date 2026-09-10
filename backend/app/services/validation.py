"""
Image validation service.

Every image is inspected before it is allowed into the processing
pipeline. Failures are reported explicitly (sensor, stage, reason) --
the pipeline never continues silently on a broken input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.models.schemas import ImageValidationResult


def validate_image(sensor: str, path: Optional[Path]) -> ImageValidationResult:
    if path is None:
        return ImageValidationResult(
            sensor=sensor,
            valid=False,
            reason=(
                f"No image file found for sensor {sensor} in its data directory. "
                f"Place a .png/.jpg/.tif file at backend/data/{sensor}/."
            ),
        )

    if not path.exists():
        return ImageValidationResult(sensor=sensor, valid=False, path=str(path),
                                      reason="File path resolved but does not exist on disk.")

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception as exc:  # pragma: no cover - defensive
        return ImageValidationResult(sensor=sensor, valid=False, path=str(path),
                                      reason=f"File could not be decoded: {exc}")

    if img is None:
        return ImageValidationResult(
            sensor=sensor, valid=False, path=str(path),
            reason="File exists but is not a readable/valid image (corrupt or unsupported format).",
        )

    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]

    if h < 32 or w < 32:
        return ImageValidationResult(
            sensor=sensor, valid=False, path=str(path), width=w, height=h, channels=channels,
            reason=f"Image resolution too small ({w}x{h}) for reliable feature-based registration.",
        )

    if img.dtype != np.uint8:
        # Not a failure by itself, but noted -- normalization handles this downstream.
        pass

    return ImageValidationResult(
        sensor=sensor, valid=True, path=str(path), width=w, height=h, channels=channels,
    )

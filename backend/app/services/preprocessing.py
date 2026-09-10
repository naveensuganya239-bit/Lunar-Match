"""
Sensor-aware preprocessing.

Different sensors (optical vs. radar vs. infrared) need different
photometric handling before feature detection will work well across
them. This module normalizes contrast and denoises while preserving
structural (edge) content that feature detectors rely on.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.services.analysis import to_gray


def preprocess(img: np.ndarray, sensor: str) -> np.ndarray:
    """
    Convert to a normalized single-channel float-free uint8 image ready
    for feature detection. Steps are chosen based on generic
    cross-sensor imaging differences (illumination, contrast, noise)
    rather than being hard-coded per sensor name.
    """
    gray = to_gray(img)

    # Denoise mildly -- SAR in particular exhibits speckle noise, and
    # infrared (IIRS)-style imagery is often low-contrast/noisy.
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # CLAHE equalizes local contrast so illumination differences between
    # sensors/passes do not dominate the gradient signal used by feature
    # detectors -- this is the key step that makes cross-sensor matching
    # feasible at all.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)

    return equalized


def gradient_representation(gray_img: np.ndarray) -> np.ndarray:
    """
    Edge/gradient-domain representation. Optional secondary
    representation useful when direct intensity matching across
    sensors (e.g. optical vs radar) fails, since edges/structure are
    more consistent across modalities than raw intensity.
    """
    sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobel_x, sobel_y)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag

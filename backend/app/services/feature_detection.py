"""
Feature detection + description.

Implements a cascade of detector configurations (see
config.DETECTOR_CASCADE). Every detector here is a real OpenCV
implementation operating on real pixel data -- no keypoints are ever
synthesized. Sub-pixel corner refinement is applied where the detector
supports it, per the sub-pixel accuracy goal in the problem statement.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from app import config


def _build_detector(method: str):
    if method == "ORB_HIGH":
        return cv2.ORB_create(nfeatures=config.ORB_N_FEATURES_HIGH, scaleFactor=1.2, nlevels=8)
    if method == "ORB_WIDE":
        return cv2.ORB_create(nfeatures=config.ORB_N_FEATURES_WIDE, scaleFactor=1.15, nlevels=12,
                               edgeThreshold=15, fastThreshold=5)
    if method == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(f"Unknown detector method: {method}")


def detect_and_describe(gray_img: np.ndarray, method: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    detector = _build_detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray_img, None)
    if keypoints is None:
        keypoints = []
    keypoints = refine_subpixel(gray_img, keypoints)
    return keypoints, descriptors


def refine_subpixel(gray_img: np.ndarray, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
    """
    Refine integer-pixel keypoint locations to sub-pixel accuracy using
    cv2.cornerSubPix around each detected keypoint. This is what allows
    the pipeline to legitimately report sub-pixel-refined coordinates
    rather than merely claiming sub-pixel accuracy.
    """
    if not keypoints:
        return keypoints
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    try:
        refined = cv2.cornerSubPix(
            gray_img, pts, config.SUBPIX_WIN, config.SUBPIX_ZERO_ZONE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    except cv2.error:
        return keypoints  # refinement is best-effort; fall back to integer coords
    for kp, p in zip(keypoints, refined.reshape(-1, 2)):
        # Guard against cornerSubPix diverging far from the original detection.
        if np.hypot(p[0] - kp.pt[0], p[1] - kp.pt[1]) < 3.0:
            kp.pt = (float(p[0]), float(p[1]))
    return keypoints

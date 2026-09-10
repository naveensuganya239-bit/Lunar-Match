"""
Quantitative evaluation of a registration result.

Every metric here is computed directly from the actual inlier
correspondences and the estimated transformation -- nothing is
hard-coded. See config.CONFIDENCE_THRESHOLDS for the (documented,
adjustable) thresholds used to classify confidence.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from app import config


def compute_rmse(src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray) -> Tuple[float, float]:
    """
    Reprojection error of inlier source points transformed by H against
    their matched reference points. Returns (rmse_px, max_error_px).
    """
    ones = np.ones((src_pts.shape[0], 1), dtype=np.float64)
    homo = np.hstack([src_pts, ones])
    projected = (H @ homo.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - dst_pts, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_err = float(np.max(errors)) if len(errors) else 0.0
    return rmse, max_err


def classify_confidence(inliers: int, inlier_ratio: float, rmse: float) -> str:
    thr = config.CONFIDENCE_THRESHOLDS
    high = thr["HIGH"]
    med = thr["MEDIUM"]
    if inliers >= high["min_inliers"] and inlier_ratio >= high["min_inlier_ratio"] and rmse <= high["max_rmse"]:
        return "HIGH"
    if inliers >= med["min_inliers"] and inlier_ratio >= med["min_inlier_ratio"] and rmse <= med["max_rmse"]:
        return "MEDIUM"
    return "LOW"


def information_preservation_check(
    ref_shape, warped_mask: np.ndarray, inlier_ratio: float, rmse: float
) -> Dict:
    """
    Heuristic, measurement-based assessment of whether the warp likely
    preserved useful information from the source image, based on
    (a) how much of the reference frame the warped source actually
    covers, and (b) registration quality metrics themselves.
    """
    h, w = ref_shape[:2]
    coverage = float(np.count_nonzero(warped_mask)) / float(h * w)

    issues = []
    if coverage < 0.15:
        issues.append("Warped source covers a small fraction of the reference frame; "
                       "much of the reference area has no corresponding source information.")
    if inlier_ratio < 0.15:
        issues.append("Low inlier ratio suggests the estimated transform may not "
                       "generalize well across the full image (possible local distortion).")
    if rmse > 6.0:
        issues.append("High reprojection RMSE indicates possible misalignment / geometric "
                       "distortion beyond acceptable tolerance.")

    status = "GOOD" if not issues else ("DEGRADED" if len(issues) == 1 else "POOR")
    return {
        "status": status,
        "coverage_fraction": round(coverage, 4),
        "issues": issues,
    }

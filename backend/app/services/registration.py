"""
Registration orchestration for a single (reference, source) sensor pair.

Runs the adaptive detector cascade (config.DETECTOR_CASCADE), matches
features, robustly estimates a homography with RANSAC, warps the
source image into the reference frame, and computes real quantitative
metrics. Never fabricates a result: if every method in the cascade
fails to produce enough inliers, the pair is reported as FAILED with
an explicit reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from app import config
from app.services import evaluation, feature_detection, feature_matching, preprocessing


@dataclass
class RegistrationResult:
    sensor: str
    status: str
    method_used: Optional[str] = None
    keypoints_reference: int = 0
    keypoints_source: int = 0
    candidate_matches: int = 0
    valid_matches: int = 0
    inlier_matches: int = 0
    inlier_ratio: float = 0.0
    rmse_px: Optional[float] = None
    max_reprojection_error_px: Optional[float] = None
    transformation_type: Optional[str] = None
    homography: Optional[np.ndarray] = None
    confidence: Optional[str] = None
    warped_image: Optional[np.ndarray] = None
    warped_mask: Optional[np.ndarray] = None
    information_preservation: Optional[dict] = None
    failure_reason: Optional[str] = None
    failure_stage: Optional[str] = None
    recovery_attempted: List[str] = field(default_factory=list)
    match_viz_points: Optional[dict] = None  # for frontend visualization


def register_pair(ref_gray_proc: np.ndarray, src_gray_proc: np.ndarray,
                   ref_shape, src_original: np.ndarray, sensor_name: str) -> RegistrationResult:
    result = RegistrationResult(sensor=sensor_name, status="FAILED")
    attempted = []

    for method in config.DETECTOR_CASCADE:
        attempted.append(method)
        kp_ref, desc_ref = feature_detection.detect_and_describe(ref_gray_proc, method)
        kp_src, desc_src = feature_detection.detect_and_describe(src_gray_proc, method)

        if len(kp_ref) < 4 or len(kp_src) < 4:
            continue

        matches = feature_matching.match_descriptors(desc_ref, desc_src)

        if len(matches) < config.MIN_MATCH_COUNT:
            continue

        dst_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        src_pts = np.float32([kp_src[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD, maxIters=5000, confidence=0.995
        )

        if H is None or mask is None:
            continue

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())

        if n_inliers < config.MIN_MATCH_COUNT:
            continue

        # Sanity-check the homography: reject degenerate/near-singular transforms.
        det = np.linalg.det(H[:2, :2])
        if not np.isfinite(det) or abs(det) < 1e-6:
            continue

        inlier_src = src_pts[inlier_mask]
        inlier_dst = dst_pts[inlier_mask]
        rmse, max_err = evaluation.compute_rmse(inlier_src, inlier_dst, H)

        h_ref, w_ref = ref_shape[:2]
        warped = cv2.warpPerspective(src_original, H, (w_ref, h_ref))
        warped_gray = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_mask = (warped_gray > 0).astype(np.uint8)

        inlier_ratio = n_inliers / max(len(matches), 1)
        confidence = evaluation.classify_confidence(n_inliers, inlier_ratio, rmse)
        info_pres = evaluation.information_preservation_check((h_ref, w_ref), warped_mask, inlier_ratio, rmse)

        result.status = "SUCCESS"
        result.method_used = method
        result.keypoints_reference = len(kp_ref)
        result.keypoints_source = len(kp_src)
        result.candidate_matches = len(matches)
        result.valid_matches = len(matches)
        result.inlier_matches = n_inliers
        result.inlier_ratio = round(inlier_ratio, 4)
        result.rmse_px = round(rmse, 3)
        result.max_reprojection_error_px = round(max_err, 3)
        result.transformation_type = "Homography (projective, RANSAC)"
        result.homography = H
        result.confidence = confidence
        result.warped_image = warped
        result.warped_mask = warped_mask
        result.information_preservation = info_pres
        result.recovery_attempted = attempted[:-1]
        result.match_viz_points = {
            "ref_points": inlier_dst[:250].tolist(),
            "src_points": inlier_src[:250].tolist(),
            "all_ref_points": dst_pts[:400].tolist(),
            "all_src_points": src_pts[:400].tolist(),
            "inlier_flags": inlier_mask[:400].tolist(),
        }
        return result

    # Every method in the cascade failed.
    result.recovery_attempted = attempted
    result.failure_stage = "FEATURE_MATCHING" if attempted else "FEATURE_DETECTION"
    result.failure_reason = (
        "Insufficient reliable feature correspondences were found between the reference "
        f"and source image after trying {len(attempted)} detector configuration(s) "
        f"({', '.join(attempted)}). This can happen when illumination, scale, or sensor "
        "modality differences are too large for local feature descriptors to bridge."
    )
    return result

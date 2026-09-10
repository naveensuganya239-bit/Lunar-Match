"""
Feature matching.

Brute-force Hamming matching (binary descriptors: ORB/AKAZE) with
Lowe's ratio test + mutual cross-check to reject ambiguous
correspondences before they ever reach RANSAC. All match objects are
real cv2.DMatch results referencing real detected keypoints -- indices
are never hand-picked.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2

from app import config


def match_descriptors(desc_ref, desc_src, norm_type=cv2.NORM_HAMMING) -> List[cv2.DMatch]:
    if desc_ref is None or desc_src is None or len(desc_ref) < 2 or len(desc_src) < 2:
        return []

    bf = cv2.BFMatcher(norm_type, crossCheck=False)

    # Ratio test: reference -> source
    knn_rs = bf.knnMatch(desc_ref, desc_src, k=2)
    good_rs = {}
    for pair in knn_rs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < config.LOWE_RATIO * n.distance:
            good_rs[(m.queryIdx, m.trainIdx)] = m

    # Ratio test the other direction for a mutual/cross check
    knn_sr = bf.knnMatch(desc_src, desc_ref, k=2)
    good_sr = set()
    for pair in knn_sr:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < config.LOWE_RATIO * n.distance:
            good_sr.add((m.trainIdx, m.queryIdx))  # (ref_idx, src_idx)

    mutual = [m for key, m in good_rs.items() if key in good_sr]
    mutual.sort(key=lambda m: m.distance)
    return mutual

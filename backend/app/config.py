"""
Central configuration for the Lunar Multi-Sensor Registration backend.

All paths and tunable parameters live here so they are not scattered
across modules. Values can be overridden with environment variables,
which keeps secrets and deployment-specific values out of source code.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/

# ---------------------------------------------------------------------------
# Dataset / storage locations
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("LUNAR_DATA_DIR", BASE_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("LUNAR_OUTPUT_DIR", BASE_DIR / "outputs"))
HISTORY_FILE = OUTPUT_DIR / "run_history.json"

# Frontend static files (served by the backend in production)
FRONTEND_DIR = Path(os.getenv("LUNAR_FRONTEND_DIR", BASE_DIR.parent / "frontend"))

SENSOR_DIRS = {
    "OHRC": DATA_DIR / "OHRC",
    "TMC": DATA_DIR / "TMC",
    "IIRS": DATA_DIR / "IIRS",
    "SAR": DATA_DIR / "SAR",
}

REFERENCE_SENSOR = "OHRC"
SOURCE_SENSORS = ["TMC", "IIRS", "SAR"]

for d in list(SENSOR_DIRS.values()) + [OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Storage backend abstraction
# ---------------------------------------------------------------------------
# "local" (default, filesystem under OUTPUT_DIR) or "s3" (requires the
# corresponding environment variables; see services/storage.py).
STORAGE_BACKEND = os.getenv("LUNAR_STORAGE_BACKEND", "local")
S3_BUCKET = os.getenv("LUNAR_S3_BUCKET", "")
S3_REGION = os.getenv("LUNAR_S3_REGION", "")
# AWS credentials are read from the standard AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY environment variables by boto3 -- never hard-coded.

# ---------------------------------------------------------------------------
# Registration pipeline parameters (engineering decisions, tunable)
# ---------------------------------------------------------------------------
# Ordered list of detector/matcher strategies attempted, in order, until one
# produces a geometrically valid registration. This implements the adaptive
# multi-method strategy required by the spec (Part 30 of the brief).
DETECTOR_CASCADE = ["ORB_HIGH", "ORB_WIDE", "AKAZE"]

MIN_MATCH_COUNT = 8          # below this, geometric estimation is not attempted
RANSAC_REPROJ_THRESHOLD = 5.0  # px, used by cv2.findHomography
LOWE_RATIO = 0.75            # Lowe's ratio test threshold
ORB_N_FEATURES_HIGH = 6000
ORB_N_FEATURES_WIDE = 12000

CONFIDENCE_THRESHOLDS = {
    # (min_inliers, min_inlier_ratio, max_rmse_px) -> label
    "HIGH": {"min_inliers": 40, "min_inlier_ratio": 0.35, "max_rmse": 3.0},
    "MEDIUM": {"min_inliers": 15, "min_inlier_ratio": 0.18, "max_rmse": 6.0},
}

# Sub-pixel keypoint refinement (cv2.cornerSubPix) window
SUBPIX_WIN = (5, 5)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (3, 30, 0.001)  # cv2.TERM_CRITERIA_EPS + COUNT, max_iter, eps

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
CORS_ORIGINS = os.getenv("LUNAR_CORS_ORIGINS", "*").split(",")

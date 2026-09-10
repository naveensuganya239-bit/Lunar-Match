"""
Generates a SYNTHETIC demonstration dataset standing in for real
Chandrayaan-2 OHRC/TMC/IIRS/SAR imagery.

IMPORTANT / HONESTY NOTE
-------------------------
No real ISRO/Chandrayaan-2 sensor imagery was available to this
implementation (none was supplied as input, and this environment has
no network access to ISRO/PRADAN data archives). To allow the
registration pipeline to be demonstrated end-to-end with genuine,
non-fabricated computer-vision processing, this script procedurally
generates a single synthetic "ground truth" lunar-like terrain (craters,
ridges, regolith texture) and then derives four sensor-style renditions
of it with realistic geometric and photometric differences:

  OHRC : the sharpest, highest-resolution rendition (reference)
  TMC  : mildly blurred / slightly different crop+rotation+scale
         (simulates a wider-swath mapping camera)
  IIRS : contrast-inverted-ish / pseudo-infrared palette, geometric
         offset (simulates a different spectral sensor)
  SAR  : speckle noise + rotation + scale, edge-dominant appearance
         (simulates radar backscatter imagery)

All four images depict the SAME underlying synthetic scene, so genuine
feature detection/matching/RANSAC registration across them is
meaningful -- but they are NOT real lunar sensor data. Replace the
files in backend/data/<SENSOR>/ with real OHRC/TMC/IIRS/SAR imagery to
use this system on actual mission data; no code changes are required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import SENSOR_DIRS  # noqa: E402

RNG = np.random.default_rng(42)


def generate_base_terrain(size=1400) -> np.ndarray:
    """Procedural crater-like grayscale terrain used as ground truth."""
    terrain = np.zeros((size, size), dtype=np.float32)

    # Low-frequency rolling terrain via summed random gaussians (cheap Perlin-ish noise)
    for _ in range(6):
        sigma = RNG.uniform(size * 0.05, size * 0.25)
        amp = RNG.uniform(5, 20)
        cx, cy = RNG.uniform(0, size, 2)
        y, x = np.mgrid[0:size, 0:size]
        terrain += amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))

    terrain += 120  # baseline albedo

    # Craters: bright rim, dark floor -- distinct enough to give real, detectable corners/edges.
    n_craters = 55
    for _ in range(n_craters):
        r = RNG.uniform(8, 70)
        cx, cy = RNG.uniform(r, size - r, 2)
        y, x = np.mgrid[0:size, 0:size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        floor = -RNG.uniform(15, 35) * np.exp(-(dist ** 2) / (2 * (r * 0.6) ** 2))
        rim = RNG.uniform(10, 25) * np.exp(-((dist - r) ** 2) / (2 * (r * 0.18) ** 2))
        terrain += floor + rim

    # Fine regolith texture (real noise, not decoration -- gives detectors something to lock onto)
    fine_noise = RNG.normal(0, 6, (size, size)).astype(np.float32)
    fine_noise = cv2.GaussianBlur(fine_noise, (3, 3), 0)
    terrain += fine_noise

    # A handful of linear ridge/fault-like structures
    for _ in range(4):
        pt1 = tuple(RNG.integers(0, size, 2))
        pt2 = tuple(RNG.integers(0, size, 2))
        overlay = np.zeros_like(terrain)
        cv2.line(overlay, pt1, pt2, color=float(RNG.uniform(10, 18)), thickness=int(RNG.integers(3, 7)))
        overlay = cv2.GaussianBlur(overlay, (9, 9), 0)
        terrain += overlay

    terrain = cv2.normalize(terrain, None, 0, 255, cv2.NORM_MINMAX)
    return terrain.astype(np.uint8)


def make_ohrc(base: np.ndarray) -> np.ndarray:
    # Sharpen slightly to emulate a high-resolution optical reference.
    sharpened = cv2.addWeighted(base, 1.4, cv2.GaussianBlur(base, (0, 0), 3), -0.4, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _similarity_warp(img: np.ndarray, angle_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)


def make_tmc(base: np.ndarray) -> np.ndarray:
    warped = _similarity_warp(base, angle_deg=6, scale=0.97, tx=15, ty=-10)
    blurred = cv2.GaussianBlur(warped, (5, 5), 1.2)
    # Slightly different illumination (gamma shift) to simulate a different pass/sun angle.
    gamma = 1.15
    norm = (blurred / 255.0) ** gamma
    return (norm * 255).astype(np.uint8)


def make_iirs(base: np.ndarray) -> np.ndarray:
    warped = _similarity_warp(base, angle_deg=-4, scale=1.03, tx=-20, ty=18)
    # Pseudo-infrared palette effect: strong local-contrast change + mild inversion of shading
    inv_ish = cv2.addWeighted(warped, 0.55, 255 - warped, 0.45, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    return clahe.apply(inv_ish)


def make_sar(base: np.ndarray) -> np.ndarray:
    warped = _similarity_warp(base, angle_deg=12, scale=1.08, tx=25, ty=25)
    edges = cv2.Laplacian(warped, cv2.CV_32F, ksize=3)
    edge_mag = cv2.normalize(np.abs(edges), None, 0, 255, cv2.NORM_MINMAX)
    sar_like = cv2.addWeighted(warped.astype(np.float32), 0.5, edge_mag, 0.5, 0)
    # Speckle noise, characteristic of radar imagery.
    speckle = sar_like * (1 + RNG.normal(0, 0.18, sar_like.shape))
    return np.clip(speckle, 0, 255).astype(np.uint8)


def main():
    base = generate_base_terrain()

    ohrc = make_ohrc(base)
    tmc = make_tmc(base)
    iirs = make_iirs(base)
    sar = make_sar(base)

    outputs = {"OHRC": ohrc, "TMC": tmc, "IIRS": iirs, "SAR": sar}
    for sensor, img in outputs.items():
        target_dir = SENSOR_DIRS[sensor]
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{sensor.lower()}_demo.png"
        cv2.imwrite(str(out_path), img)
        print(f"Wrote {out_path} ({img.shape[1]}x{img.shape[0]})")

    print("\nSynthetic demo dataset generated. See this script's docstring for the "
          "honesty note: replace these files with real sensor imagery to use real data.")


if __name__ == "__main__":
    main()

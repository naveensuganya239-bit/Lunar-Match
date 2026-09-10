# Lunar Multi-Sensor Image Registration

A working prototype that registers TMC (Terrain Mapping Camera), IIRS
(Imaging Infrared Spectrometer), and SAR (Synthetic Aperture Radar) lunar
source images against an OHRC (Orbiter High Resolution Camera) reference
image, using real OpenCV feature detection / matching / RANSAC geometric
estimation, computes real quantitative metrics (RMSE, inlier ratio,
inlier count), fuses the results into a multi-sensor product, and
generates a confidence-aware "Lunar AI" scientific interpretation.

Built against the SIH problem statement: *generic software for finding
correspondence between Chandrayaan-2 acquired optical images and lunar
reference images with sub-pixel-oriented accuracy, plus evaluation
metrics.*

---

## 1. Read this first -- honesty notes

- **No real Chandrayaan-2 imagery was supplied or available to this
  build.** The attached PDF is the problem specification, not an image
  dataset, and this environment has no network access to ISRO/PRADAN
  archives. `backend/scripts/generate_demo_dataset.py` procedurally
  generates a synthetic crater-like terrain and derives four
  sensor-style renditions of it (see the script's docstring). This lets
  the pipeline be demonstrated end-to-end with **genuine, unmodified
  computer-vision processing** -- but the demo images are not real
  lunar data. **Drop real OHRC/TMC/IIRS/SAR files into
  `backend/data/<SENSOR>/` and no code changes are needed** -- the
  system will process them exactly the same way.
- Every metric shown in the UI (feature counts, match counts, inlier
  ratio, RMSE, confidence) is computed live from the actual images in
  that run. Nothing is hard-coded or animated independently of a real
  API response.
- "Lunar AI" is a structured, confidence-aware rule-based interpretation
  engine (see `app/services/lunar_ai.py`) -- not a general chatbot. It
  optionally uses the Anthropic API purely to rephrase its own findings
  into fluent prose (if `ANTHROPIC_API_KEY` is set); with no key
  configured it works fully using the deterministic engine.
- Geolocation (lat/long), mineral composition, and geological age are
  **never** produced -- this system has no means of measuring them, and
  the AI module explicitly labels this as a limitation rather than
  guessing.

---

## 2. Architecture

```
Frontend (static HTML/CSS/JS, dark space "mission control" theme)
        |  fetch() JSON + image URLs
        v
Backend (FastAPI)
  services/dataset.py         -- locate sensor files on disk
  services/validation.py      -- real file/format/size checks
  services/analysis.py        -- measured image characteristics
  services/preprocessing.py   -- CLAHE + denoise (sensor-aware)
  services/feature_detection.py -- ORB/AKAZE cascade + sub-pixel refine
  services/feature_matching.py  -- ratio test + mutual cross-check
  services/registration.py    -- RANSAC homography, adaptive retry cascade
  services/evaluation.py      -- RMSE, inlier ratio, confidence, info-preservation
  services/fusion.py          -- sensor-aware multi-sensor composite
  services/lunar_ai.py        -- confidence-aware scientific interpretation
  services/storage.py         -- storage abstraction (local now, S3-ready)
  services/pipeline.py        -- orchestrates all of the above end-to-end
```

No stage is a placeholder; every stage is real, runnable OpenCV/NumPy code.

---

## 3. Directory structure

```
lunar-registration/
  backend/
    app/
      main.py, config.py
      api/routes.py
      models/schemas.py
      services/  (see above)
      utils/logging_config.py
    data/OHRC/ TMC/ IIRS/ SAR/     <- put real sensor images here
    outputs/                       <- generated run outputs (gitignored)
    scripts/generate_demo_dataset.py
    tests/test_pipeline.py
    requirements.txt
    .env.example
  frontend/
    index.html, styles.css, app.js
  run_backend.sh
  run_frontend.sh
  README.md
```

---

## 4. Installation & running

Requires Python 3.10+.

```bash
# 1. Backend (also auto-generates the demo dataset on first run)
./run_backend.sh
# -> API on http://localhost:8000, docs at http://localhost:8000/docs

# 2. Frontend, in a second terminal
./run_frontend.sh
# -> UI on http://localhost:5173
```

Manual equivalent:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/generate_demo_dataset.py   # only needed once, or skip if you added real images
uvicorn app.main:app --reload

cd ../frontend
python3 -m http.server 5173
```

If your frontend is served from somewhere other than `localhost`, set
`window.LUNAR_API_BASE` at the top of `frontend/app.js` (or inject it
before `app.js` loads) to point at your backend URL.

---

## 5. Dataset setup

Place exactly one image per sensor in:

```
backend/data/OHRC/  <- reference (required)
backend/data/TMC/
backend/data/IIRS/
backend/data/SAR/
```

Accepted formats: `.png .jpg .jpeg .tif .tiff .bmp`. If a folder has
multiple files, the first (alphabetically) is used. Missing/corrupt
files are reported explicitly via `/api/dataset/status`, never silently
skipped.

---

## 6. API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness check |
| GET | `/api/dataset/status` | validation status of all 4 sensor files |
| POST | `/api/run?mode=quick_demo` | run full pipeline with TMC+IIRS+SAR |
| POST | `/api/run?mode=custom&sources=TMC&sources=SAR` | run with a subset of sources (OHRC always included as reference) |
| GET | `/api/history` | last 50 runs (id, timestamp, sensors, status) |
| GET | `/api/download/{run_id}/{filename}` | download a specific output file |
| GET | `/api/report/{run_id}` | download the full JSON reproducibility report |
| static | `/outputs/{run_id}/{filename}` | direct image access (used by `<img>` tags) |

`POST /api/run` response shape (abridged):

```jsonc
{
  "run_id": "a1b2c3d4e5", "timestamp": "...", "status": "SUCCESS",
  "validation": {...}, "characteristics": {...},
  "registration": {
    "TMC": {
      "status": "SUCCESS", "method_used": "ORB_HIGH",
      "keypoints_reference": 5211, "keypoints_source": 4980,
      "candidate_matches": 430, "inlier_matches": 419,
      "inlier_ratio": 0.974, "rmse_px": 1.881,
      "confidence": "HIGH", "information_preservation": {...},
      "match_viz_points": { "ref_points": [...], "src_points": [...] }
    },
    "IIRS": {...}, "SAR": {...}
  },
  "fusion_output_path": "a1b2c3d4e5/lunar_multisensor_fusion.png",
  "ai_analysis": { "overall_confidence": "HIGH", "observations": [...], "limitations": [...] }
}
```

---

## 7. Registration algorithm

For each (OHRC, source) pair:

1. **Preprocess** -- grayscale, bilateral denoise, CLAHE local-contrast
   normalization (this is what makes cross-sensor matching feasible at
   all despite illumination/modality differences).
2. **Adaptive detector cascade** -- tries `ORB_HIGH` -> `ORB_WIDE` ->
   `AKAZE` in order, stopping at the first configuration that yields a
   geometrically valid result (`config.DETECTOR_CASCADE`). Keypoints are
   refined to sub-pixel locations with `cv2.cornerSubPix`.
3. **Matching** -- brute-force Hamming matching with Lowe's ratio test
   (0.75) applied in both directions and mutually cross-checked, which
   rejects ambiguous correspondences before RANSAC ever sees them.
4. **Robust estimation** -- `cv2.findHomography` with RANSAC
   (reprojection threshold 5.0 px), plus a determinant sanity check to
   reject degenerate transforms.
5. **Warping** -- `cv2.warpPerspective` of the full-resolution source
   image into the OHRC frame.
6. If any step yields too few correspondences, the next cascade method
   is tried; if all methods fail, the sensor is marked `FAILED` with an
   explicit stage + reason -- it is excluded from fusion but the rest of
   the run continues.

---

## 8. Evaluation metrics (all computed, never hard-coded)

- **Inlier match count / candidate match count / inlier ratio** -- directly
  from the RANSAC mask.
- **RMSE (px)** -- reprojection error of inlier source points transformed
  by the estimated homography against their matched reference points
  (`evaluation.compute_rmse`).
- **Confidence (HIGH/MEDIUM/LOW)** -- thresholded on inlier count, inlier
  ratio, and RMSE together (`config.CONFIDENCE_THRESHOLDS`); documented,
  adjustable, and never silently overridden to look better.
- **Information preservation** -- checks warped-source coverage of the
  reference frame plus the above metrics, and reports `GOOD / DEGRADED /
  POOR` with the specific issue(s) found.

---

## 9. Lunar AI

`lunar_ai.build_analysis()` receives the *actual* registration results
(status, confidence, inlier ratio, RMSE, information-preservation) for
every sensor and produces:

- Per-finding **OBSERVED / DERIVED / UNCERTAIN** classification.
- An **overall confidence** rolled up from the individual sensor
  confidences, which gates how strongly the narrative is allowed to
  read.
- Explicit **sensor-contribution** descriptions (what OHRC/TMC/IIRS/SAR
  each physically measure).
- Explicit **limitations** (no geolocation, no composition, no age --
  every run, unconditionally).
- Optional fluent-prose enhancement via the Anthropic API, constrained
  to only rephrase the findings already computed -- never to add new
  ones -- with silent, safe fallback to the rule-based text if no API
  key is set or the call fails.

---

## 10. UI workflow

1. **Landing** -- sensor status cards (OHRC visually marked FIXED
   REFERENCE), "Run analysis" (quick demo: all 3 sources) or "Custom
   sensor selection" (choose any subset of TMC/IIRS/SAR; OHRC can't be
   removed).
2. **Processing** -- live pipeline stage list.
3. **Results** -- large fused output image + download buttons; per-sensor
   metric cards; feature-correspondence canvas (toggle inliers-only vs.
   all candidates, per sensor pair); before/after/fused compare tabs;
   Lunar AI panel with OBSERVED/DERIVED/UNCERTAIN tags; run history.

---

## 11. Testing

```bash
cd backend
source .venv/bin/activate
pytest -q
```

`tests/test_pipeline.py` runs the real pipeline against the bundled
dataset (no mocking) and asserts: all sensors validate, the full
`quick_demo` run succeeds with positive inlier counts, and that a
missing reference image produces an explicit `FAILED` result rather
than a silent one. All 4 tests pass against the bundled demo dataset.

---

## 12. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Frontend shows "backend unreachable" | Backend not running, or CORS/port mismatch -- check `run_backend.sh` output and `LUNAR_CORS_ORIGINS`. |
| A sensor shows `FAILED` with "insufficient reliable feature correspondences" | Real, expected pipeline behavior on genuinely hard image pairs (large modality/scale gap). Try supplying higher-resolution or better-overlapping imagery. |
| `ModuleNotFoundError: cv2` | Run `pip install -r requirements.txt` inside the backend virtualenv. |
| Empty/black fused output | No source sensor registered successfully this run; check the metric cards' failure reasons. |

---

## 13. Demo script (3-5 minutes)

1. Open the frontend -- point out OHRC marked as the fixed reference and
   TMC/IIRS/SAR as sources, with live validation status per sensor.
2. Click **Run analysis** -- narrate the pipeline stages as they light up.
3. On the results screen: show the large fused output, then the metric
   cards -- read out real inlier counts / RMSE / confidence per sensor.
4. Open the feature-correspondence view, switch between "inliers only"
   and "all candidate matches" for a sensor pair -- show real line
   correspondences, not decoration.
5. Open before/after/fused compare tabs for one sensor.
6. Show the Lunar AI panel -- point out the OBSERVED vs. DERIVED vs.
   UNCERTAIN tags and the explicit limitations section.
7. Download the output image and the JSON report.

---

## 14. Technical limitations (honest)

- Demo dataset is synthetic (see Section 1) -- not validated against
  real Chandrayaan-2 imagery.
- Sub-pixel *refinement* is implemented (`cv2.cornerSubPix`); true
  validated sub-pixel *registration accuracy* would require a
  ground-truth control-point set to measure against, which isn't
  available here -- the README and UI do not claim sub-pixel accuracy
  has been proven, only that refinement is applied.
- Fusion is a designed structural-contribution composite, not a
  radiometrically rigorous multi-sensor data fusion product; it is
  intended for visual/scientific triage, not quantitative photometry.
- No authentication/multi-user concerns are handled -- this is a
  single-user local prototype.
- S3 storage backend is defined as an interface (`services/storage.py`)
  but not implemented; local filesystem storage is used today.

## 15. Future improvements

- Real Chandrayaan-2 OHRC/TMC/IIRS/SAR dataset validation.
- Ground-truth control points to quantitatively validate sub-pixel
  accuracy claims.
- S3/GCS backend implementation behind the existing `StorageService`
  interface.
- ROI (region-of-interest) auto-detection and click-to-highlight, per
  the original brief's Part 19 -- not implemented in this pass; the
  data needed (a validated feature/structure classifier) is out of
  scope for a registration-focused prototype and should be added as a
  follow-on module rather than guessed at.

---

## 16. Requirement audit (abridged)

| Requirement | Status | Where |
|---|---|---|
| OHRC fixed reference, TMC/IIRS/SAR sources | Done | `config.py`, enforced in `pipeline.py`, visually enforced in UI |
| Real dataset loading, no fake images | Done (synthetic demo data, clearly labeled) | `services/dataset.py`, `scripts/generate_demo_dataset.py` |
| Image validation with explicit failure reporting | Done | `services/validation.py` |
| Image characteristics analysis | Done | `services/analysis.py` |
| Sensor-aware preprocessing | Done | `services/preprocessing.py` |
| Real feature detection/matching, no fabricated points | Done | `services/feature_detection.py`, `feature_matching.py` |
| Adaptive multi-method registration cascade | Done | `config.DETECTOR_CASCADE`, `registration.py` |
| Real RANSAC-based geometric registration + warping | Done | `registration.py` |
| Sub-pixel keypoint refinement | Done (refinement implemented; accuracy not independently validated -- stated honestly) | `feature_detection.refine_subpixel` |
| Quantitative evaluation (RMSE, inlier ratio, inlier count) computed, not hard-coded | Done | `evaluation.py` |
| Information-preservation check | Done | `evaluation.information_preservation_check` |
| Multi-sensor fusion (not naive blend) | Done | `fusion.py` |
| Lunar AI, confidence-aware, OBSERVED/DERIVED/UNCERTAIN | Done | `lunar_ai.py` |
| No hallucinated geolocation/composition/age | Done (explicitly withheld) | `lunar_ai.py` limitations |
| Explicit error handling, no silent failure | Done | `registration.py` failure paths, `routes.py` |
| Reproducibility record per run | Done | `pipeline._append_history`, `run_record.json` |
| Quick Demo + Custom modes | Done | `api/routes.py`, frontend landing view |
| Feature-match visualization (inliers/outliers toggle) | Done | frontend `drawMatchCanvas` |
| Before/after/overlay comparison | Done | frontend compare tabs |
| Downloadable output + report | Done | `/api/download`, `/api/report` |
| Cloud-ready storage abstraction | Done (interface only; S3 not implemented) | `services/storage.py` |
| Processing history | Done | `/api/history`, frontend history list |
| Dark space "mission control" UI | Done | `frontend/styles.css`, `index.html` |
| ROI explorer with clickable regions | Not implemented | See Section 15 |

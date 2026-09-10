"""
End-to-end pipeline orchestrator.

Implements the full chain required by the specification:

  dataset -> validation -> characteristics analysis -> preprocessing ->
  feature detection -> feature matching -> robust registration ->
  quantitative evaluation -> information preservation check ->
  multi-sensor fusion -> Lunar AI interpretation -> reproducibility record

Every stage's real output feeds the next stage. Nothing here is a stub;
if a stage fails for a given sensor, that sensor is marked FAILED with
an explicit reason and excluded from fusion, while the rest of the run
continues.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from app import config
from app.services import analysis, dataset, fusion, lunar_ai, preprocessing, registration, storage, validation


def _imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def run_pipeline(mode: str, selected_sources: List[str]) -> Dict:
    run_id = uuid.uuid4().hex[:10]
    timestamp = datetime.now(timezone.utc).isoformat()
    store = storage.get_storage_service()

    sensors_to_load = [config.REFERENCE_SENSOR] + [s for s in selected_sources if s in config.SOURCE_SENSORS]
    paths = dataset.discover_dataset(sensors_to_load)

    # ---- Stage 1: validation ------------------------------------------------
    validations = {s: validation.validate_image(s, p) for s, p in paths.items()}

    ref_validation = validations[config.REFERENCE_SENSOR]
    if not ref_validation.valid:
        return {
            "run_id": run_id, "timestamp": timestamp, "mode": mode,
            "sensors_used": sensors_to_load, "reference_sensor": config.REFERENCE_SENSOR,
            "validation": [v.model_dump() for v in validations.values()],
            "characteristics": [], "registration": [],
            "fusion_output_path": None, "ai_analysis": None,
            "status": "FAILED",
            "failure_reason": f"Reference sensor OHRC failed validation: {ref_validation.reason}",
        }

    ref_img = _imread(Path(ref_validation.path))
    ref_gray_raw = analysis.to_gray(ref_img)

    # ---- Stage 2: characteristics analysis ---------------------------------
    characteristics = {config.REFERENCE_SENSOR: analysis.analyze_image(config.REFERENCE_SENSOR, ref_img)}

    # ---- Stage 3: preprocessing (reference) --------------------------------
    ref_gray_proc = preprocessing.preprocess(ref_img, config.REFERENCE_SENSOR)

    registration_results: Dict[str, registration.RegistrationResult] = {}
    registered_for_fusion = {}

    for sensor in [s for s in selected_sources if s in config.SOURCE_SENSORS]:
        v = validations.get(sensor)
        if v is None or not v.valid:
            registration_results[sensor] = registration.RegistrationResult(
                sensor=sensor, status="FAILED",
                failure_stage="VALIDATION",
                failure_reason=v.reason if v else "Sensor not requested/found.",
            )
            continue

        src_img = _imread(Path(v.path))
        characteristics[sensor] = analysis.analyze_image(sensor, src_img)
        src_gray_proc = preprocessing.preprocess(src_img, sensor)

        result = registration.register_pair(
            ref_gray_proc, src_gray_proc, ref_gray_raw.shape, src_img, sensor
        )
        registration_results[sensor] = result

        if result.status == "SUCCESS":
            registered_for_fusion[sensor] = {
                "image": result.warped_image,
                "mask": result.warped_mask,
                "confidence": result.confidence,
            }

    # ---- Fusion -------------------------------------------------------------
    fusion_rel_path = None
    if registered_for_fusion:
        fusion_out = fusion.fuse(ref_gray_raw, registered_for_fusion)
        fusion_rel_path = f"{run_id}/lunar_multisensor_fusion.png"
        store.save_image(fusion_out["fused_image"], fusion_rel_path)
    else:
        # Nothing registered successfully -- still expose the OHRC reference
        # itself as the "output" so the user always has something concrete,
        # clearly not claiming any fusion took place.
        fusion_rel_path = f"{run_id}/ohrc_reference_only.png"
        store.save_image(ref_img, fusion_rel_path)

    # Save registered images individually for before/after/overlay viewing.
    per_sensor_paths = {}
    for sensor, res in registration_results.items():
        if res.status == "SUCCESS":
            rel = f"{run_id}/{sensor}_registered.png"
            store.save_image(res.warped_image, rel)
            per_sensor_paths[sensor] = rel

    # ---- Lunar AI -------------------------------------------------------------
    ai_input = {
        s: {
            "status": r.status,
            "confidence": r.confidence,
            "inlier_matches": r.inlier_matches,
            "valid_matches": r.valid_matches,
            "inlier_ratio": r.inlier_ratio,
            "rmse_px": r.rmse_px,
            "transformation_type": r.transformation_type,
            "information_preservation": r.information_preservation,
            "failure_reason": r.failure_reason,
        }
        for s, r in registration_results.items()
    }
    ai_analysis = lunar_ai.build_analysis(ai_input)

    # ---- Reproducibility record ----------------------------------------------
    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": mode,
        "reference_sensor": config.REFERENCE_SENSOR,
        "sensors_used": sensors_to_load,
        "reference_path": ref_validation.path,
        "source_paths": {s: (v.path if v.valid else None) for s, v in validations.items() if s != config.REFERENCE_SENSOR},
        "validation": {s: v.model_dump() for s, v in validations.items()},
        "characteristics": {s: c.model_dump() for s, c in characteristics.items()},
        "registration": {
            s: {k: v for k, v in vars(r).items()
                if k not in ("homography", "warped_image", "warped_mask", "match_viz_points")}
            for s, r in registration_results.items()
        },
        "fusion_output_path": fusion_rel_path,
        "per_sensor_registered_paths": per_sensor_paths,
        "ai_analysis": ai_analysis,
        "status": "SUCCESS" if any(r.status == "SUCCESS" for r in registration_results.values()) else "PARTIAL_FAILURE",
    }

    _append_history(record)
    store.save_json(json.dumps(record, indent=2, default=str), f"{run_id}/run_record.json")

    # Build the API-facing response, including match visualization points.
    response = dict(record)
    response["registration"] = {
        s: {**record["registration"][s], "match_viz_points": r.match_viz_points}
        for s, r in registration_results.items()
    }
    return response


def _append_history(record: Dict) -> None:
    history = []
    if config.HISTORY_FILE.exists():
        try:
            history = json.loads(config.HISTORY_FILE.read_text())
        except Exception:
            history = []
    summary = {
        "run_id": record["run_id"],
        "timestamp": record["timestamp"],
        "mode": record["mode"],
        "sensors_used": record["sensors_used"],
        "status": record["status"],
        "fusion_output_path": record["fusion_output_path"],
    }
    history.insert(0, summary)
    config.HISTORY_FILE.write_text(json.dumps(history[:50], indent=2))


def get_history() -> List[Dict]:
    if not config.HISTORY_FILE.exists():
        return []
    try:
        return json.loads(config.HISTORY_FILE.read_text())
    except Exception:
        return []

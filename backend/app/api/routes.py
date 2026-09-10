from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app import config
from app.services import dataset, pipeline, storage, validation

router = APIRouter()


@router.get("/dataset/status")
def dataset_status():
    """Report which sensor images are currently present on disk (Part 7/8)."""
    sensors = [config.REFERENCE_SENSOR] + config.SOURCE_SENSORS
    paths = dataset.discover_dataset(sensors)
    results = []
    for s in sensors:
        v = validation.validate_image(s, paths[s])
        results.append({**v.model_dump(), "role": "REFERENCE" if s == config.REFERENCE_SENSOR else "SOURCE"})
    return {"sensors": results}


@router.post("/run")
def run_analysis(
    mode: str = Query("quick_demo", pattern="^(quick_demo|custom)$"),
    sources: Optional[List[str]] = Query(None),
):
    """
    Run the full registration + fusion + Lunar AI pipeline.

    mode=quick_demo -> uses all of TMC, IIRS, SAR as sources (OHRC always reference).
    mode=custom     -> uses only the sensors listed in `sources` (OHRC always reference,
                        cannot be removed).
    """
    if mode == "quick_demo":
        selected = list(config.SOURCE_SENSORS)
    else:
        selected = [s for s in (sources or []) if s in config.SOURCE_SENSORS]
        if not selected:
            raise HTTPException(400, "Custom mode requires at least one of TMC, IIRS, SAR in `sources`.")

    try:
        result = pipeline.run_pipeline(mode=mode, selected_sources=selected)
    except Exception as exc:  # top-level safety net -- never return a bare 500 with no info
        raise HTTPException(500, f"Pipeline execution failed: {exc}") from exc

    return result


@router.get("/history")
def run_history():
    return {"runs": pipeline.get_history()}


@router.get("/download/{run_id}/{filename}")
def download_output(run_id: str, filename: str):
    store = storage.get_storage_service()
    rel_path = f"{run_id}/{filename}"
    full_path = store.get_download_path(rel_path)
    if full_path is None:
        raise HTTPException(404, f"Output file not found: {rel_path}")
    return FileResponse(str(full_path), filename=filename)


@router.get("/report/{run_id}")
def download_report(run_id: str):
    """Serve the JSON reproducibility/analysis record as the 'analysis report' download."""
    report_path = config.OUTPUT_DIR / run_id / "run_record.json"
    if not report_path.exists():
        raise HTTPException(404, "Report not found for this run_id.")
    return FileResponse(str(report_path), filename=f"lunar_registration_report_{run_id}.json",
                         media_type="application/json")

"""
Tests run against the actual on-disk dataset (backend/data/*) using the
real pipeline -- no mocks of the CV code. Run with:  pytest -q

Expected results with the bundled synthetic demo dataset:
  - Dataset validation passes for all four sensors.
  - TMC, IIRS, SAR all register successfully with a non-trivial number
    of inliers (the synthetic images share real structure with OHRC).
  - Fusion output file is produced.
  - Lunar AI analysis includes at least one OBSERVED finding per sensor.

If you replace the demo dataset with real, harder sensor imagery, some
sensors may legitimately FAIL to register -- that is a correct pipeline
behavior, not a test bug. Adjust assertions accordingly for your data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import dataset, pipeline, validation  # noqa: E402
from app import config  # noqa: E402


def test_dataset_files_present():
    for sensor in [config.REFERENCE_SENSOR] + config.SOURCE_SENSORS:
        path = dataset.find_sensor_image(sensor)
        assert path is not None, f"No demo image found for {sensor}; run scripts/generate_demo_dataset.py"


def test_all_sensors_validate():
    for sensor in [config.REFERENCE_SENSOR] + config.SOURCE_SENSORS:
        path = dataset.find_sensor_image(sensor)
        result = validation.validate_image(sensor, path)
        assert result.valid, result.reason


def test_full_pipeline_quick_demo():
    result = pipeline.run_pipeline("quick_demo", config.SOURCE_SENSORS)
    assert result["status"] in ("SUCCESS", "PARTIAL_FAILURE")
    assert result["fusion_output_path"]
    assert "ai_analysis" in result

    for sensor in config.SOURCE_SENSORS:
        reg = result["registration"][sensor]
        # With the bundled synthetic dataset these should succeed; if you
        # swapped in real, more difficult imagery this may legitimately
        # differ -- inspect reg["failure_reason"] in that case.
        assert reg["status"] == "SUCCESS", f"{sensor}: {reg.get('failure_reason')}"
        assert reg["inlier_matches"] > 0
        assert reg["rmse_px"] is not None and reg["rmse_px"] >= 0


def test_failure_is_explicit_when_reference_missing(tmp_path, monkeypatch):
    # Point OHRC's directory at an empty temp dir to exercise the explicit
    # failure path (Part 23: never fail silently).
    empty_dir = tmp_path / "OHRC_empty"
    empty_dir.mkdir()
    monkeypatch.setitem(config.SENSOR_DIRS, "OHRC", empty_dir)
    result = pipeline.run_pipeline("quick_demo", config.SOURCE_SENSORS)
    assert result["status"] == "FAILED"
    assert "OHRC" in result["failure_reason"]

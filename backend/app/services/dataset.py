"""
Dataset loading service.

Locates the actual sensor image files on disk. OHRC is always resolved
as the fixed reference; TMC/IIRS/SAR are resolved as source images.
No image is ever substituted, generated, or faked here -- if a file is
missing, that is reported as a real validation failure, not papered over.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from app.config import SENSOR_DIRS

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_sensor_image(sensor: str) -> Optional[Path]:
    """Return the first valid image file found in a sensor's data directory."""
    directory = SENSOR_DIRS.get(sensor)
    if directory is None or not directory.exists():
        return None
    candidates = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    return candidates[0] if candidates else None


def discover_dataset(sensors) -> Dict[str, Optional[Path]]:
    """Discover the on-disk path (or None) for every requested sensor."""
    return {sensor: find_sensor_image(sensor) for sensor in sensors}

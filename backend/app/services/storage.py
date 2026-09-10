"""
Storage abstraction.

Image-processing code never touches the filesystem or a cloud SDK
directly -- it goes through StorageService, which today implements a
local-filesystem backend and defines the interface a future S3 /
GCS / Azure Blob backend would implement. This keeps the pipeline
cloud-deployment-ready without adding cloud dependencies to the
prototype.

No credentials are ever hard-coded; an S3 backend would read
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / LUNAR_S3_BUCKET from the
environment (see app/config.py).
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app import config


class StorageService(abc.ABC):
    @abc.abstractmethod
    def save_image(self, image: np.ndarray, relative_path: str) -> str:
        ...

    @abc.abstractmethod
    def save_json(self, data: str, relative_path: str) -> str:
        ...

    @abc.abstractmethod
    def get_download_path(self, relative_path: str) -> Optional[Path]:
        ...


class LocalStorageService(StorageService):
    def __init__(self, root: Path = config.OUTPUT_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_image(self, image: np.ndarray, relative_path: str) -> str:
        full_path = self.root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(Path(relative_path).suffix or ".png", image)
        if not ok:
            raise IOError(f"Failed to encode image for {relative_path}")
        full_path.write_bytes(buf.tobytes())
        return str(full_path)

    def save_json(self, data: str, relative_path: str) -> str:
        full_path = self.root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(data, encoding="utf-8")
        return str(full_path)

    def get_download_path(self, relative_path: str) -> Optional[Path]:
        p = self.root / relative_path
        return p if p.exists() else None


def get_storage_service() -> StorageService:
    if config.STORAGE_BACKEND == "s3":
        raise NotImplementedError(
            "S3 backend is not wired into this prototype. Implement an S3StorageService "
            "here using boto3 + config.S3_BUCKET/S3_REGION, and switch LUNAR_STORAGE_BACKEND=s3."
        )
    return LocalStorageService()

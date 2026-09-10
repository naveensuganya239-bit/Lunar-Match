"""Pydantic response/request schemas shared across the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImageValidationResult(BaseModel):
    sensor: str
    valid: bool
    path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    reason: Optional[str] = None


class ImageCharacteristics(BaseModel):
    sensor: str
    width: int
    height: int
    channels: int
    mean_intensity: float
    std_intensity: float
    contrast_michelson: float
    edge_density: float
    estimated_feature_density: str
    dynamic_range: int
    notes: List[str] = Field(default_factory=list)


class RegistrationMetrics(BaseModel):
    sensor: str
    status: str  # SUCCESS | FAILED
    method_used: Optional[str] = None
    keypoints_reference: Optional[int] = None
    keypoints_source: Optional[int] = None
    candidate_matches: Optional[int] = None
    valid_matches: Optional[int] = None
    inlier_matches: Optional[int] = None
    inlier_ratio: Optional[float] = None
    rmse_px: Optional[float] = None
    max_reprojection_error_px: Optional[float] = None
    transformation_type: Optional[str] = None
    confidence: Optional[str] = None
    information_preservation: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None
    failure_stage: Optional[str] = None
    recovery_attempted: Optional[List[str]] = None
    ai_interpretation_enabled: bool = True


class RunResult(BaseModel):
    run_id: str
    timestamp: str
    mode: str
    sensors_used: List[str]
    reference_sensor: str
    validation: List[ImageValidationResult]
    characteristics: List[ImageCharacteristics]
    registration: List[RegistrationMetrics]
    fusion_output_path: Optional[str] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    status: str

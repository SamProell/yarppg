"""Utilities for ROI (region of interest) detection and manipulation."""

from typing import Callable

from .detector import RoiDetector
from .facemesh_segmenter import FaceMeshDetector
from .region_of_interest import contour_to_mask, overlay_mask, pixelate, pixelate_mask
from .selfie_segmenter import SelfieDetector

detectors: dict[str, Callable[..., RoiDetector]] = {
    "facemesh": FaceMeshDetector,
    "selfie": SelfieDetector,
}

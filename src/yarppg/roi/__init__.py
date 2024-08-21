"""Utilities for ROI (region of interest) detection and manipulation."""

from .detector import RoiDetector
from .facemesh_segmenter import FaceMeshDetector
from .region_of_interest import (
    RegionOfInterest,
    contour_to_mask,
    overlay_mask,
    pixelate,
    pixelate_mask,
)
from .selfie_segmenter import SelfieDetector

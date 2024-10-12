"""Utilities for ROI (region of interest) detection and manipulation.

yarPPG comes with a number of ROI detectors, which find the face in the
input frame and provide a mask with the relevant region(s). The following
detectors are currently implemented:

- [`FaceMeshDetector`][yarppg.FaceMeshDetector] (default) - uses MediaPipe's
  FaceMesh landmarker.
- [`SelfieDetector`][yarppg.SelfieDetector] - uses MediaPipe's SelfieSegmenter
  solution. Selfie segmentation is slower than FaceMesh and may not work in a
  real-time application.

Detectors return a [`RegionOfInterest`][yarppg.RegionOfInterest] container
that stores the original image, the ROI mask and an optional background mask.
"""

from typing import Callable

from .detector import RoiDetector
from .facemesh_segmenter import FaceMeshDetector
from .roi_tools import contour_to_mask, overlay_mask, pixelate, pixelate_mask
from .selfie_segmenter import SelfieDetector

detectors: dict[str, Callable[..., RoiDetector]] = {
    "facemesh": FaceMeshDetector,
    "selfie": SelfieDetector,
}

"""Provides the base class of the ROI detector."""

import numpy as np

from ..containers import RegionOfInterest


class RoiDetector:
    """Base class for ROI detectors."""

    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        """Find region of interest in the given frame."""
        raise NotImplementedError("Detect method needs to be overwritten.")

    def __call__(self, frame: np.ndarray) -> RegionOfInterest:
        """Apply detector on the given frame."""
        return self.detect(frame)

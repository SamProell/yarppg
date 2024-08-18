"""Provides the base class of the ROI detector."""

import numpy as np

from .region_of_interest import RegionOfInterest


class ROIDetector:
    """Base class for ROI detectors."""

    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        """Find region of interest in the given frame."""
        raise NotImplementedError("detect method needs to be overwritten.")

    def __call__(self, frame: np.ndarray) -> RegionOfInterest:
        """Apply detector on the given frame."""
        return self.detect(frame)

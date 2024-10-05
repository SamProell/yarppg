"""Defines some containers passed between objects of the yarppg application."""

from dataclasses import dataclass

import numpy as np

from .roi import RegionOfInterest


@dataclass
class Color:
    """Defines a color in RGB(A) format."""

    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def null(cls):
        """Create empty color with NaN values."""
        return cls(np.nan, np.nan, np.nan)

    def __array__(self):
        return np.array([self.r, self.g, self.b, self.a])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Convert numpy array to `Color` object."""
        if len(arr) in {3, 4} and arr.ndim == 1:
            return cls(*arr)
        raise ValueError(f"Cannot interpret {arr=!r}")


@dataclass
class RppgResult:
    """Container for rPPG computation results."""

    value: float
    """Output value of the rPPG signal extractor."""
    roi: RegionOfInterest
    """Region of interest identified in the current frame."""
    roi_mean: Color
    """Mean color of the ROI."""
    bg_mean: Color
    """Mean color of the background."""
    hr: float = np.nan
    """Heart rate estimate in frames per beat."""

    def __array__(self):
        return np.asarray([self.value])

"""Base processor for rPPG signal computation."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..roi.region_of_interest import RegionOfInterest


@dataclass
class Color:
    """Defines a color in RGB(A) format."""

    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def null(cls):
        return cls(np.nan, np.nan, np.nan)

    def __array__(self):
        return np.array([self.r, self.g, self.b, self.a])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        if len(arr) in {3, 4} and arr.ndim == 1:
            return cls(*arr)
        raise ValueError(f"Cannot interpret {arr=!r}")


def masked_average(frame: np.ndarray, mask: np.ndarray) -> Color:
    """Calculate average color of the masked region."""
    r, g, b, a = cv2.mean(frame, mask)
    return Color(r, g, b, a)


@dataclass
class RppgResult:
    """Container for rPPG computation results."""

    value: float
    roi: RegionOfInterest
    roi_mean: Color
    bg_mean: Color

    def __array__(self):
        return np.asarray([self.value])


class Processor:
    """Default rPPG processor."""

    def process(self, frame: np.ndarray, roi: RegionOfInterest):
        """Calculate average green channel in the roi area."""
        avg = masked_average(frame, roi.mask)
        bg_mean = Color.null()
        if roi.bg_mask is not None:
            bg_mean = masked_average(frame, roi.bg_mask)

        return RppgResult(avg.g, roi, roi_mean=avg, bg_mean=bg_mean)

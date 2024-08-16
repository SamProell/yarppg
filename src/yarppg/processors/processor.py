from dataclasses import dataclass

import cv2
import numpy as np

from ..roi.region_of_interest import RegionOfInterest


@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def null(cls):
        return cls(np.nan, np.nan, np.nan)


def masked_average(frame: np.ndarray, mask: np.ndarray) -> Color:
    """Calculate average color of the masked region."""
    r, g, b, a = cv2.mean(frame, mask)
    return Color(r, g, b, a)


@dataclass
class RppgResult:
    value: float
    roi: RegionOfInterest
    roi_mean: Color
    bg_mean: Color


class Processor:
    def process(self, frame: np.ndarray, roi: RegionOfInterest):
        avg = masked_average(frame, roi.mask)
        bg_mean = Color.null()
        if roi.bg_mask is not None:
            bg_mean = masked_average(frame, roi.bg_mask)

        return RppgResult(avg.g, roi, roi_mean=avg, bg_mean=bg_mean)

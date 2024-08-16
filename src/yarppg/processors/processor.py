from dataclasses import dataclass

import cv2
import numpy as np

from ..roi.region_of_interest import RegionOfInterest


def calculate_color_means(
    frame: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    r, g, b, _ = cv2.mean(frame, mask)
    return r, g, b


@dataclass
class Color:
    r: float
    g: float
    b: float

    @classmethod
    def null(cls):
        return cls(np.nan, np.nan, np.nan)


@dataclass
class RppgResult:
    value: float
    roi: RegionOfInterest
    roi_mean: Color
    bg_mean: Color


class Processor:
    def process(self, frame: np.ndarray, roi: RegionOfInterest):
        r, g, b = calculate_color_means(frame, roi.mask)
        if roi.bg_mask is None:
            bg_mean = Color.null()
        else:
            bg_mean = Color(*calculate_color_means(frame, roi.bg_mask))
        return RppgResult(g, roi, roi_mean=Color(r, g, b), bg_mean=bg_mean)

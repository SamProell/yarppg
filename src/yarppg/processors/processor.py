"""Base processor for rPPG signal computation."""

import cv2
import numpy as np

from ..digital_filter import DigitalFilter
from ..roi.region_of_interest import RegionOfInterest
from ..rppg_result import Color, RppgResult


def masked_average(frame: np.ndarray, mask: np.ndarray) -> Color:
    """Calculate average color of the masked region."""
    if mask.sum() == 0:
        return Color.null()
    r, g, b, a = cv2.mean(frame, mask)
    return Color(r, g, b, a)


class Processor:
    """Default rPPG processor."""

    def process(self, frame: np.ndarray, roi: RegionOfInterest):
        """Calculate average green channel in the roi area."""
        avg = masked_average(frame, roi.mask)
        bg_mean = Color.null()
        if roi.bg_mask is not None:
            bg_mean = masked_average(frame, roi.bg_mask)

        return RppgResult(avg.g, roi, roi_mean=avg, bg_mean=bg_mean)

    def reset(self):
        """Reset internal state and intermediate values."""
        pass  # no persistent values in base class


class FilteredProcessor(Processor):
    """Processor with temporal filtering of the extracted signal."""

    def __init__(self, processor: Processor, livefilter: DigitalFilter | None = None):
        self.processor = processor
        self.livefilter = livefilter

    def process(self, frame: np.ndarray, roi: RegionOfInterest):
        """Calculate processor output and apply digital filter."""
        result = self.processor.process(frame, roi)
        if self.livefilter is not None and np.isfinite(result.value):
            # only calculate filter update if not NaN
            result.value = self.livefilter.process(result.value)
        return result

    def reset(self):
        """Reset internal state and intermediate values."""
        self.processor.reset()
        if self.livefilter is not None:
            self.livefilter.reset()

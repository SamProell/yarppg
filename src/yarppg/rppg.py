"""Provides the RPPG orchestrator class."""
import numpy as np

from . import helpers, hr_calculator, processors, roi
from .rppg_result import RppgResult


class Rppg:
    """Orchestrator for complete rPPG pipeline.

    Args:
       roi_detector: detector for identifying the region of interest (and background).
       processor: rPPG signal extraction algorithm.
       hr_calc: heart rate calculation algorithm.
    """

    def __init__(
        self,
        roi_detector: roi.ROIDetector,
        processor: processors.Processor,
        hr_calc: hr_calculator.HrCalculator,
    ):
        self.roi_detector = roi_detector
        self.processor = processor
        self.hr_calculator = hr_calc

        self.history: list[RppgResult] = []

    def process_frame(self, frame: np.ndarray) -> RppgResult:
        """Process a single frame from video or live stream."""
        roi = self.roi_detector.detect(frame)
        result = self.processor.process(frame, roi)
        result.hr = self.hr_calculator.update(result)

        self.history.append(result)
        return result

    def process_video(self, filename: str) -> list[RppgResult]:
        """Convenience function to process an entire video file at once."""
        results = []
        for frame in helpers.frames_from_video(filename):
            results.append(self.process_frame(frame))
        return results

    def reset(self) -> None:
        """Reset processor and history."""
        self.history.clear()
        self.processor.reset()

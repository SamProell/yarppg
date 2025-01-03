"""Provides the Rppg orchestrator class.

The orchestrator ties together the typical steps required in an rPPG pipeline:

1. region of interest (ROI) identification ([yarppg.roi][])
2. rPPG signal extraction ([yarppg.processors][])
3. heart rate estimation ([yarppg.hr_calculator][])

`Rppg`'s [`process_frame`][yarppg.Rppg.process_frame] method performs the three
steps from above in order and produces an [yarppg.containers.RppgResult][] that
holds the extracted rPPG signal value as well as the frame, ROI and some
additional information.

```python
import yarppg

default_settings = yarppg.Settings()
rppg = yarppg.Rppg.from_settings(default_settings)

result = rppg.process_frame(frame)  # input a (h x w x 3)-image array.
print(result.hr)
```

"""

import pathlib
from typing import Literal, overload

import numpy as np
import pandas as pd
import scipy.signal

from . import digital_filter, helpers, hr_calculator, processors, roi
from .containers import RppgResult
from .settings import Settings


class Rppg:
    """Orchestrator for the complete rPPG pipeline.

    If unspecified the following default configuration is used:

    - [`FaceMeshDetector`][yarppg.FaceMeshDetector] is used for ROI identification.
    - The base [`Processor`][yarppg.Processor] extracts the average green value.
    - A [`PeakBasedHrCalculator`][yarppg.PeakBasedHrCalculator] estimates HR

    Args:
       roi_detector: detector for identifying the region of interest (and background).
       processor: rPPG signal extraction algorithm.
       hr_calc: heart rate calculation algorithm.
       fps: expected frames per second of the camera/video
    """

    def __init__(
        self,
        roi_detector: roi.RoiDetector | None = None,
        processor: processors.Processor | None = None,
        hr_calc: hr_calculator.HrCalculator | None = None,
        fps: float = 30,
    ):
        self.roi_detector = roi_detector or roi.FaceMeshDetector()
        self.processor = processor or processors.Processor()
        self.hr_calculator = hr_calc or hr_calculator.PeakBasedHrCalculator(fps)

    def process_frame(self, frame: np.ndarray) -> RppgResult:
        """Process a single frame from video or live stream."""
        roi = self.roi_detector.detect(frame)
        result = self.processor.process(roi)
        result.hr = self.hr_calculator.update(result.value)

        return result

    @overload
    def process_video(self, filename: ..., as_dataframe: Literal[True]) -> pd.DataFrame:
        ...

    @overload
    def process_video(
        self, filename: ..., as_dataframe: Literal[False] = ...
    ) -> list[RppgResult]:
        ...

    def process_video(self, filename: str | pathlib.Path, as_dataframe=False):
        """Convenience function to process an entire video file at once."""
        results = []
        for frame in helpers.frames_from_video(filename):
            result = self.process_frame(frame)
            if as_dataframe:
                results.append(result.to_series())
            else:
                results.append(result)
        if as_dataframe:
            return pd.concat(results).T
        return results

    def reset(self) -> None:
        """Reset internal elements."""
        self.processor.reset()

    @classmethod
    def from_settings(cls, settings: Settings) -> "Rppg":
        """Instantiate rPPG orchestrator with the given settings."""
        detector = roi.detectors[settings.detector]()
        processor = processors.algorithms[settings.algorithm]()
        if settings.filter:
            if settings.filter == "bandpass":
                b, a = scipy.signal.iirfilter(2, [0.7, 1.8], fs=30, btype="band")
                livefilter = digital_filter.DigitalFilter(b, a)
            else:
                livefilter = digital_filter.make_digital_filter(settings.filter)
            processor = processors.FilteredProcessor(processor, livefilter)
        return cls(detector, processor)

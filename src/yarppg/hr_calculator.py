"""Heart rate calculation utilities."""

from collections import deque

import numpy as np
import scipy.signal

from .rppg_result import RppgResult


class HrCalculator:
    """Base class for heart rate calculation."""

    def update(self, data: RppgResult) -> float:  # noqa: ARG002
        """Process the new data and update HR estimate."""
        return np.nan


class PeakBasedHrCalculator(HrCalculator):
    """Peak-based heart rate calculation."""

    def __init__(
        self,
        fs: float,
        window_seconds: float = 10,
        distance: float = 0.5,
    ):
        self.winsize = int(fs * window_seconds)
        self.values = deque(maxlen=self.winsize)
        self.mindist = int(fs * distance)

    def update(self, data: RppgResult) -> float:
        """Process the new data and update HR estimate."""
        self.values.append(data.value)
        if len(self.values) < self.winsize:
            return np.nan
        peaks, _ = scipy.signal.find_peaks(self.values, distance=self.mindist)
        return np.diff(peaks).mean()

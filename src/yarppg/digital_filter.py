"""Provides tools for applying digital filters in a real-time application."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.signal


@dataclass
class FilterConfig:
    """Container for configuration of a digital filter.

    This configuration allows creation of filters through `scipy.signal.iirfilter`.

    Attributes:
        fs: expected sampling rate of the signal.
        f1: first cut-off frequency.
        f2: seconds cut-off frequency. Required for bandpass filters. Defaults to None.
        btype: type of the filter. low-, high-, or bandpass.
        ftype: type of the filter design. Butterworth is good for most cases.
        order: order of the filter.
    """

    fs: float
    f1: float
    f2: float | None = None
    btype: str = "low"
    ftype: str = "butter"
    order: int = 2


class DigitalFilter:
    """Live digital filter processing one sample at a time.

    Args:
        b: numerator coefficients obtained from scipy.
        a: denominator coefficients obtained from scipy.
        xi: first signal value used to initialize the filter state.
    """

    def __init__(self, b: np.ndarray, a: np.ndarray, xi: float = 0):
        self.b = b
        self.a = a
        self.reset(xi)

    def process(self, x: float) -> float:
        """Process incoming data and update filter state."""
        y, self.zi = scipy.signal.lfilter(self.b, self.a, [x], zi=self.zi)
        return y[0]

    def process_signal(self, x: Sequence[float]) -> np.ndarray:
        """Process an entire signal at once (SciPy's lfilter with current state)."""
        y, self.zi = scipy.signal.lfilter(self.b, self.a, x, zi=self.zi)
        return y

    def reset(self, xi: float = 0):
        """Reset filter state to initial value."""
        self.zi = scipy.signal.lfiltic(self.b, self.a, [xi], xi)


def filtercoeffs_from_config(cfg: FilterConfig):
    """Get coefficients (b, a) for filter with given settings."""
    cutoff = [cfg.f1]
    if cfg.f2:
        cutoff.append(cfg.f2)
    b, a = scipy.signal.iirfilter(
        cfg.order, cutoff, btype=cfg.btype, ftype=cfg.ftype, fs=cfg.fs
    )
    return b, a


def make_digital_filter(cfg: FilterConfig) -> DigitalFilter:
    """Create live digital filter with given settings."""
    b, a = filtercoeffs_from_config(cfg)
    return DigitalFilter(b, a)

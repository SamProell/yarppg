"""Provides tools for applying digital filters in a real-time application."""

from typing import Sequence

import numpy as np
import scipy.signal


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

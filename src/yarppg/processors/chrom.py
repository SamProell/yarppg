"""Chrominance-based rPPG method introduced by de Haan et al. [^1].

[^1]: de Haan, G., & Jeanne, V. (2013). Robust Pulse Rate From
    Chrominance-Based rPPG. IEEE Transactions on Biomedical Engineering,
    60(10), 2878-2886. https://doi.org/10.1109/TBME.2013.2266196
"""

from typing import Literal

import numpy as np

from ..roi.region_of_interest import RegionOfInterest
from .processor import Color, Processor, RppgResult


class ChromProcessor(Processor):
    """Chrominance-based rPPG algorithm by de Haan & Jeanne (2013).

    Args:
        winsize: window size for moving average calculations. Defaults to 45.
        method: method to use. Can be 'xovery' or 'fixed'. Defaults to "xovery".
    """

    def __init__(
        self, winsize: int = 45, method: Literal["fixed", "xovery"] = "xovery"
    ):
        Processor.__init__(self)

        self.winsize = winsize
        self.method = method

        self._rgbs: list[Color] = []
        self._xs: list[float] = []
        self._ys: list[float] = []

    def process(self, frame: np.ndarray, roi: RegionOfInterest) -> RppgResult:
        """Calculate pulse signal update according to Chrom algorithm."""
        result = super().process(frame, roi)
        self._rgbs.append(result.roi_mean)

        if self.method == "fixed":
            result.value = self._calculate_fixed_update()

        elif self.method == "xovery":
            result.value = self._calculate_xovery_update()

        return result

    def _calculate_fixed_update(self) -> float:
        rgbmean = Color.from_array(np.mean(self._rgbs[-self.winsize :], axis=0))

        rn = self._rgbs[-1].r / (rgbmean.r or 1.0)
        gn = self._rgbs[-1].g / (rgbmean.g or 1.0)
        bn = self._rgbs[-1].b / (rgbmean.b or 1.0)

        self._xs.append(3 * rn - 2 * gn)
        self._ys.append(1.5 * rn + gn - 1.5 * bn)

        return self._xs[-1] / (self._ys[-1] or 1.0) - 1

    def _calculate_xovery_update(self) -> float:
        rgb = self._rgbs[-1]

        self._xs.append(rgb.r - rgb.g)
        self._ys.append(0.5 * rgb.r + 0.5 * rgb.g - rgb.b)

        xmean = np.mean(self._xs[-self.winsize :])
        ymean = np.mean(self._ys[-self.winsize :])

        return float(xmean / (ymean or 1) - 1)

    def reset(self):
        """Reset internal state and intermediate values."""
        self._rgbs.clear()
        self._xs.clear()
        self._ys.clear()

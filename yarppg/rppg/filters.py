"""Provides functionality for digital signal filtering."""
import dataclasses
from typing import Optional

import numpy as np
import scipy.signal


@dataclasses.dataclass
class FilterConfig:
    fs: float
    f1: float
    f2: Optional[float] = None
    btype: str = "low"
    ftype: str = "butter"
    order: int = 2


def get_butterworth_filter(f, cutoff, btype="low", order=2):
    """Create live version of Butterworth filter."""
    b, a = scipy.signal.butter(N=order, Wn=np.divide(cutoff, f / 2.0), btype=btype)
    return DigitalFilter(b, a)


class DigitalFilter:
    def __init__(self, b, a):
        self._bs = b
        self._as = a
        self._xs = [0] * len(b)
        self._ys = [0] * (len(a) - 1)

    def process(self, x):
        if np.isnan(x):  # ignore nans, and return as is
            return x

        self._xs.insert(0, x)
        self._xs.pop()
        y = np.dot(self._bs, self._xs) / self._as[0] - np.dot(self._as[1:], self._ys)
        self._ys.insert(0, y)
        self._ys.pop()
        return y

    def __call__(self, x):
        return self.process(x)


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


if __name__ == "__main__":
    import pyqtgraph as pg

    fs = 30
    x = np.arange(0, 10, 1.0 / fs)
    y = np.sin(2 * np.pi * x) + 0.2 * np.random.normal(size=len(x))

    b, a = filtercoeffs_from_config(FilterConfig(fs, 3))
    yfilt = scipy.signal.lfilter(b, a, y)
    myfilt = DigitalFilter(b, a)
    yfilt2 = np.array([myfilt(v) for v in y])

    print(np.mean(np.abs(yfilt - yfilt2)))

    app = pg.mkQApp()
    p = pg.plot(title="test")
    p.plot(x, y)
    p.plot(x, yfilt, pen=(0, 3))
    p.plot(x, yfilt2, pen=(1, 3))
    app.exec_()

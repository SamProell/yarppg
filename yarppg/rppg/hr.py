import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import scipy.signal


def bpm_from_inds(inds, ts):
    if len(inds) < 2:
        return np.nan

    return 60. / np.mean(np.diff(ts[inds]))


def get_f(ts):
    return 1. / np.mean(np.diff(ts))


def from_peaks(vs, ts, mindist=0.35):
    if len(ts) != len(vs) or len(ts) < 2:
        return np.nan
    f = get_f(ts)
    peaks, _ = scipy.signal.find_peaks(vs, distance=int(f*mindist))

    return bpm_from_inds(peaks, ts)


class HRCalculator(QObject):
    new_hr = pyqtSignal(float)

    def __init__(self, parent=None, update_interval=30, winsize=300):
        QObject.__init__(self, parent)

        self._counter = 0
        self.update_interval = update_interval
        self.winsize = winsize

    def update(self, rppg):
        self._counter += 1
        if self._counter >= self.update_interval:
            self._counter = 0
            ts = rppg.get_ts(self.winsize)
            vs = next(rppg.get_vs(self.winsize))
            self.new_hr.emit(from_peaks(vs, ts))

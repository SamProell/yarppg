"""Provides functionality for heart rate calculation from (rPPG) signals."""
import numpy as np
import scipy.signal
from PyQt5.QtCore import QObject, pyqtSignal


def bpm_from_inds(inds, ts):
    """Calculate heart rate (in beat/min) from indices and time vector.

    Args:
        inds (`1d array-like`): indices of heart beats
        ts (`1d array-like`): time vector corresponding to indices

    Returns:
        float: heart rate in beats per minute (bpm)
    """
    if len(inds) < 2:
        return np.nan

    return 60.0 / np.mean(np.diff(ts[inds]))


def get_sampling_rate(ts):
    """Calculate sampling rate from time vector."""
    return 1.0 / np.mean(np.diff(ts))


def from_peaks(vs, ts, mindist=0.35):
    """Calculate heart rate by finding peaks in the given signal.

    Args:
        vs (`1d array-like`): pulse wave signal
        ts (`1d array-like`): time vector corresponding to pulse signal
        mindist (float): minimum distance between peaks (in seconds)

    Returns:
        float: heart rate in beats per minute (bpm)
    """
    if len(ts) != len(vs) or len(ts) < 2:
        return np.nan
    f = get_sampling_rate(ts)
    peaks, _ = scipy.signal.find_peaks(vs, distance=int(f * mindist))

    return bpm_from_inds(peaks, ts)


def from_fft(vs, ts):
    """Calculate heart rate as most dominant frequency in pulse signal.

    Args:
        vs (`1d array-like`): pulse wave signal
        ts (`1d array-like`): time vector corresponding to pulse signal

    Returns:
        float: heart rate in beats per minute (bpm)
    """
    f = get_sampling_rate(ts)
    vf = np.fft.fft(vs)
    xf = np.linspace(0.0, f / 2.0, len(vs) // 2)
    return 60 * xf[np.argmax(np.abs(vf[: len(vf) // 2]))]


class HRCalculator(QObject):
    new_hr = pyqtSignal(float)

    def __init__(
        self, parent=None, update_interval=30, winsize=300, filt_fun=None, hr_fun=None
    ):
        QObject.__init__(self, parent)

        self._counter = 0
        self.update_interval = update_interval
        self.winsize = winsize
        self.filt_fun = filt_fun
        self.hr_fun = from_peaks
        if hr_fun is not None and callable(hr_fun):
            self.hr_fun = hr_fun

    def update(self, rppg):
        self._counter += 1
        if self._counter >= self.update_interval:
            self._counter = 0
            ts = rppg.get_ts(self.winsize)
            vs = next(rppg.get_vs(self.winsize))
            if self.filt_fun is not None and callable(self.filt_fun):
                vs = self.filt_fun(vs)
            self.new_hr.emit(self.hr_fun(vs, ts))

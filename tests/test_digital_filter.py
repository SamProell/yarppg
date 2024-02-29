import numpy as np
import scipy.signal

from yarppg.rppg.filters import get_butterworth_filter


def test_process():
    fs, cutoff = 10.0, 3.0
    b, a = scipy.signal.butter(2, Wn=cutoff / fs * 2, btype="low")
    lfilter = get_butterworth_filter(fs, cutoff)

    xs = np.arange(0, 10, 0.1)
    ys = np.sin(2 * np.pi * xs) + 0.2 * np.random.normal(size=len(xs))

    yfilt = np.array([lfilter(y) for y in ys])
    yfilt_scipy = scipy.signal.lfilter(b, a, ys)

    assert np.mean(np.abs(yfilt - yfilt_scipy)) < 1e-7

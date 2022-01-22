import numpy as np
import scipy.signal

from yarppg.rppg.filters import DigitalFilter, get_butterworth_filter


def test_process():
    fs, cutoff = 10., 3.
    ba = scipy.signal.butter(2, Wn=cutoff/fs*2, btype="low")
    lfilter = get_butterworth_filter(fs, cutoff)

    xs = np.arange(0, 10, 0.1)
    ys = np.sin(2*np.pi*xs) + 0.2*np.random.normal(size=len(xs))

    yfilt = [lfilter(y) for y in ys]
    yfilt_scipy = scipy.signal.lfilter(*ba, ys)

    assert np.mean(np.abs(yfilt - yfilt_scipy)) < 1e-7

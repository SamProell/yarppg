import numpy as np
import scipy.signal

from yarppg.rppg import digital_filter


def test_filtercoeffs_unchanged():
    cfg = digital_filter.FilterConfig(10.0, 3.0)

    ba = digital_filter.filtercoeffs_from_config(cfg)
    scipy_ba = scipy.signal.iirfilter(2, cfg.f1, fs=cfg.fs, btype="low")

    assert np.array_equal(ba, scipy_ba)


def test_process():
    cfg = digital_filter.FilterConfig(10.0, f1=3.0, btype="low")
    b, a = digital_filter.filtercoeffs_from_config(cfg)
    lfilter = digital_filter.make_digital_filter(cfg)

    xs = np.arange(0, 10, 0.1)
    ys = np.sin(2 * np.pi * xs) + 0.2 * np.random.normal(size=len(xs))

    yfilt = np.array([lfilter.process(y) for y in ys])
    yfilt_scipy = scipy.signal.lfilter(b, a, ys)

    assert np.mean(np.abs(yfilt - yfilt_scipy)) < 1e-7

import pathlib

import numpy as np

import yarppg


def test_process_video(testfiles_root: pathlib.Path):
    filename = testfiles_root / "testvideo_30fps.mp4"

    rppg = yarppg.Rppg(hr_calc=yarppg.PeakBasedHrCalculator(fs=30, window_seconds=7.5))
    results = rppg.process_video(filename)

    assert len(results) == 294
    assert abs(results[-1].hr - 60) < 1

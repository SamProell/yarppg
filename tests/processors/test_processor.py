import numpy as np

import yarppg
import yarppg.roi.roi_tools
from yarppg.processors import processor


def test_masked_average(sim_roi: yarppg.RegionOfInterest):
    assert sim_roi.bg_mask is not None
    bg_avg = yarppg.roi.roi_tools.masked_average(sim_roi.baseimg, sim_roi.bg_mask)
    roi_avg = yarppg.roi.roi_tools.masked_average(sim_roi.baseimg, sim_roi.mask)

    assert np.array_equal(bg_avg, (4, 5, 6))
    assert np.array_equal(roi_avg, (56.25, 2, 3))


def test_process(sim_roi: yarppg.RegionOfInterest):
    proc = processor.Processor()

    result = proc.process(sim_roi)

    assert result.value == 2
    assert np.array_equal(result.roi_mean, (56.25, 2, 3))
    assert np.array_equal(result.bg_mean, (4, 5, 6))

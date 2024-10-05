import numpy as np

import yarppg


def test_pixelate(sim_roi: yarppg.RegionOfInterest):
    frame = sim_roi.baseimg.copy()
    yarppg.pixelate(frame, (2, 2, 10, 10), 5)

    assert np.array_equal(frame[:2], sim_roi.baseimg[:2])
    assert np.array_equal(frame[:, :2], sim_roi.baseimg[:, :2])
    assert np.all(frame[2:4, 2:4, 0] == int(sim_roi.baseimg[2:7, 2:7, 0].mean()))


def test_contour_to_mask():
    size = (10, 10)
    points = [(2, 2), (2, 5), (5, 5)]

    mask = yarppg.roi.contour_to_mask(size, points)

    assert mask.sum() == 10
    assert mask[mask > 0].mean() == 1

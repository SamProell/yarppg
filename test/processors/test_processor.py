import numpy as np
import pytest

from yarppg.rppg.processors import Processor


def test_spatial_pooling():
    frame = np.ones((10, 10, 3))
    proc = Processor()
    for i in range(10):
        proc.spatial_pooling(i*frame, append_rgb=True)

    assert proc.spatial_pooling(frame, append_rgb=False) == (1, 1, 1)
    assert proc.spatial_pooling(np.ones((0, 0, 3))) == (np.nan,)*3

    assert len(proc._rs) == 10
    assert proc._rs[-1] == 9


def test_processor_calculate():
    proc = Processor()

    assert np.isnan(proc.calculate(None))


@pytest.mark.filterwarnings("ignore")
def test_moving_average():
    assert np.isnan(Processor.moving_average_update(None, [], 1))
    assert np.isnan(Processor.moving_average_update(None, [np.nan]*3, 2))
    assert Processor.moving_average_update(None, range(5), 4) == 2.5
    assert Processor.moving_average_update(None, range(5), 10) == 2.


def test_call():
    proc = Processor()

    assert np.isnan(proc.calculate(None))
    assert np.isnan(proc(None))
    assert len(proc.vs) == 1
    assert np.isnan(proc.vs[-1])


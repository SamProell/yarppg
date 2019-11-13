import numpy as np

from yarppg.rppg.processors import ColorMeanProcessor

frame = np.zeros((10, 10, 3))
frame[..., 0] = 1
frame[..., 1] = 2
frame[..., 2] = 3

frame2 = np.zeros((10, 10, 3))
frame2[...] = 3


def test_calculate():
    r = ColorMeanProcessor("r")
    g = ColorMeanProcessor("g")
    b = ColorMeanProcessor("b")

    assert r.calculate(frame) == 3
    assert g.calculate(frame) == 2
    assert b.calculate(frame) == 1


def test_call():

    for c, v in zip("rgb", (3, 2, 1)):
        proc = ColorMeanProcessor(c)
        assert proc(frame) == v
        assert len(proc.vs) == 1 and proc.vs[-1] == v
        assert proc(frame2) == 3
        assert len(proc.vs) == 2

        proc2 = ColorMeanProcessor(c, winsize=2)

        assert proc2(frame) == v
        assert proc2(frame2) == (v+3)/2.
        assert proc2(frame2) == 3
        assert len(proc2.vs) == 3

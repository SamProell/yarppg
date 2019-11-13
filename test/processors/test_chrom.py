import numpy as np

from yarppg.rppg.processors import ChromProcessor


frame = np.ones((5, 5, 3))
for i in range(3):
    frame[..., i] = i+1


def test_call():
    chrom = ChromProcessor(winsize=3, method="xovery")

    assert abs(chrom(frame) - (1/1.5-1)) < 0.001
    assert abs(chrom(frame) - (1/1.5-1)) < 0.001
    assert len(chrom.vs) == 2
    assert abs(chrom(frame*3) - (2./3-1)) < 0.001

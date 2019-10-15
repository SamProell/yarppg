"""Chrominance-based rPPG method introduced by de Haan et al. [1]_


.. [1] de Haan, G., & Jeanne, V. (2013). Robust Pulse Rate From
   Chrominance-Based rPPG. IEEE Transactions on Biomedical Engineering,
   60(10), 2878–2886. https://doi.org/10.1109/TBME.2013.2266196
"""

import numpy as np

from .processor import Processor


class ChromProcessor(Processor):

    def __init__(self, winsize=45, method="xovery"):
        Processor.__init__(self)

        self.winsize = winsize
        self.method = method

        self.rn, self.gn, self.bn = [], [], []
        self.rmean, self.gmean, self.bmean = 0, 0, 0

        self.n = 0

    def calculate(self, roi):
        self.n += 1
        r, g, b = self.spatial_pooling(roi, append_rgb=True)

        self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
        self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
        self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)

        rn = r / (self.rmean or 1)
        gn = g / (self.gmean or 1)
        bn = b / (self.bmean or 1)

        x = rn - gn
        y = rn/2. + gn/2. - bn
        self.vs.append(x / (y or 1))

        return self.vs[-1]

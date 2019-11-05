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

        self._xs, self._ys = [], []
        self.xmean, self.ymean = 0, 0
        self.rmean, self.gmean, self.bmean = 0, 0, 0

        self.n = 0

    def calculate(self, roi_pixels):
        self.n += 1
        r, g, b = self.spatial_pooling(roi_pixels, append_rgb=True)
        v = np.nan

        if self.method == "fixed":
            self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
            self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
            self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)
            rn = r / (self.rmean or 1.)
            gn = g / (self.gmean or 1.)
            bn = b / (self.bmean or 1.)
            self._xs.append(3*rn - 2*gn)
            self._ys.append(1.5*rn + gn - 1.5*bn)

            v = self._xs[-1] / (self._ys[-1] or 1.) - 1
        elif self.method == "xovery":
            self._xs.append(r - g)
            self._ys.append(0.5*r + 0.5*g - b)
            self.xmean = self.moving_average_update(self.xmean, self._xs, self.winsize)
            self.ymean = self.moving_average_update(self.ymean, self._ys, self.winsize)

            v = self.xmean / (self.ymean or 1) - 1

        return v

    def __str__(self):
        if self.name is None:
            return "ChromProcessor(winsize={},method={})".format(self.winsize,
                                                                 self.method)
        return self.name


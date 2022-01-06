"""Plane-Orthogonal-to-Skin (POS) algorithm introduced by Wang et al. [1]_


.. [1] Wang, W., den Brinker, A. C., Stuijk, S., and de Haan, G. (2017).
   Algorithmic Principles of Remote PPG. IEEE Transactions on Biomedical
   Engineering, 64(7), 1479–1491. https://doi.org/10.1109/TBME.2016.2609282
"""

import numpy as np

from .processor import Processor


class PosProcessor(Processor):
    def __init__(self, winsize=45):
        Processor.__init__(self)

        self.winsize = winsize

        self.hs = []
        self.rmean, self.gmean, self.bmean = 0, 0, 0

        self.n = 0

    def calculate(self, roi_pixels):
        self.n += 1
        self.spatial_pooling(roi_pixels, append_rgb=True)

        # spatial averaging
        self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
        self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
        self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)

        if self.n >= self.winsize:
            # temporal normalization
            rn = np.divide(self._rs[-self.winsize:], self.rmean or 1.)
            gn = np.divide(self._gs[-self.winsize:], self.gmean or 1.)
            bn = np.divide(self._bs[-self.winsize:], self.bmean or 1.)

            # projection
            s1 = gn - bn
            s2 = -2*rn + gn + bn

            # tuning
            h = s1 + np.nanstd(s1) / np.nanstd(s2) * s2
            self.hs.append(0.)
            self.hs[-self.winsize:] = self.hs[-self.winsize:] + (h-np.nanmean(h))
            return self.hs[-self.winsize]
        self.hs.append(0)
        return 0

    def __str__(self):
        if self.name is None:
            return "PosProcessor(winsize={})".format(self.winsize)
        return self.name

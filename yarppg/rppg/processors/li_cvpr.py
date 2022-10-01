"""This processor implements some of the features suggested by Li et al. [1]_

*work in progress* (for now, this simply returns the green channel)

The arcticle can be found here:
https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Li_Remote_Heart_Rate_2014_CVPR_paper.html

.. [1] Li, X., Chen, J., Zhao, G., &#38; Pietikainen, M. (2014). Remote
   Heart Rate Measurement From Face Videos Under Realistic Situations.
   Proceedings of the IEEE Conference on Computer Vision and Pattern
   Recognition (CVPR), 4264-4271.
"""

import numpy as np

from .processor import Processor


class LiCvprProcessor(Processor):
    def __init__(self, winsize=1):
        super().__init__()

        self.winsize = winsize
        self.h = 0

    def calculate(self, roi):
        r, g, b, bg_r, bg_g, bg_b = self.spatial_pooling(roi, background=True)
        
        step_size = 0.05
        matrix = np.matrix(bg_g)
        bg_g_Htranspose = matrix.getH()
        h = self.h + (step_size * (g - self.h * bg_g) * bg_g)/(bg_g_Htranspose * bg_g)
        self.h = h
        
        rectified_g = g - (h * bg_g)
        
        return np.array(rectified_g)[0][0]

    def __str__(self):
        if self.name is None:
            return f"LiCvprProcessor(winsize={self.winsize})"
        return self.name

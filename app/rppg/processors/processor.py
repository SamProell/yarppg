import cv2


class Processor:
    def __init__(self):
        self._rs = []
        self._gs = []
        self._bs = []

        self.vs = []

    def calculate(self, roi):
        self.vs.append(0)
        return self.vs[-1]

    def __call__(self, roi):
        return self.calculate(roi)

    def spatial_pooling(self, roi, append_rgb=False):
        b, r, g, a = cv2.mean(roi)

        if append_rgb:
            self._rs.append(r)
            self._gs.append(g)
            self._bs.append(b)

        return r, g, b

import numpy as np
import scipy.signal


def get_butterworth_filter(f, cutoff, btype="low", order=2):
    ba = scipy.signal.butter(N=order, Wn=np.divide(cutoff, f/2.), btype=btype)
    return DigitalFilter(ba[0], ba[1])


class DigitalFilter:

    def __init__(self, b, a):
        self._bs = b
        self._as = a
        self._xs = [0]*len(b)
        self._ys = [0]*(len(a)-1)

    def process(self, x):
        if np.isnan(x):  # ignore nans, and return as is
            return x

        self._xs.insert(0, x)
        self._xs.pop()
        y = (np.dot(self._bs, self._xs) / self._as[0]
             - np.dot(self._as[1:], self._ys))
        self._ys.insert(0, y)
        self._ys.pop()
        return y

    def __call__(self, x):
        return self.process(x)


if __name__ == "__main__":
    fs = 30
    x = np.arange(0, 10, 1.0/fs)
    y = np.sin(2*np.pi*x) + 0.2*np.random.normal(size=len(x))

    import pyqtgraph as pg
    app = pg.QtGui.QApplication([])
    p = pg.plot(title="test")
    p.plot(x, y)
    ba = scipy.signal.butter(2, 3/fs*2)
    yfilt = scipy.signal.lfilter(ba[0], ba[1], y)
    p.plot(x, yfilt, pen=(0, 3))

    myfilt = DigitalFilter(ba[0], ba[1])
    yfilt2 = [myfilt(v) for v in y]
    p.plot(x, yfilt2, pen=(1, 3))
    app.exec_()

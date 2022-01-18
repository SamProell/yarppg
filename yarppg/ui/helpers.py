import numpy as np
import pyqtgraph as pg
import cv2


def add_multiaxis_plot(p1, **kwargs):
    p2 = pg.ViewBox()
    p1.scene().addItem(p2)
    p1.hideAxis("right")
    p1.getAxis("right").linkToView(p2)
    p2.setXLink(p1)

    line = pg.PlotCurveItem(**kwargs)
    p2.addItem(line)

    def update_view():
        p2.setGeometry(p1.vb.sceneBoundingRect())
        p2.linkedViewChanged(p1.vb, p2.XAxis)
    update_view()
    p1.vb.sigResized.connect(update_view)

    return line, p2


def get_autorange(data, factor):
    if np.all(np.isnan(data)):
        return 0, 1
    x1, x2 = np.nanmin(data), np.nanmax(data)
    pad = (x2 - x1) * factor
    return x1 - pad, x2 + pad

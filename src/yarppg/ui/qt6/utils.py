"""Various utility functions related to the user interface."""

import numpy as np
import pyqtgraph
from numpy.typing import ArrayLike


def plain_image_item(data):
    """Create a `pyqtgraph.ImageView` showing only the actual image."""
    img_item = pyqtgraph.image(data)
    img_item.ui.histogram.hide()
    img_item.ui.roiBtn.hide()
    img_item.ui.menuBtn.hide()
    return img_item


def add_multiaxis_plot(
    p1: pyqtgraph.PlotItem, **kwargs
) -> tuple[pyqtgraph.PlotCurveItem, pyqtgraph.ViewBox]:
    """Add a new line in multiaxis view on top of the given base plot."""
    p2 = pyqtgraph.ViewBox()
    p1.scene().addItem(p2)  # type: ignore
    p1.hideAxis("right")
    p1.getAxis("right").linkToView(p2)
    p2.setXLink(p1)

    line = pyqtgraph.PlotCurveItem(**kwargs)
    p2.addItem(line)

    def update_view():
        p2.setGeometry(p1.vb.sceneBoundingRect())  # type: ignore
        p2.linkedViewChanged(p1.vb, p2.XAxis)

    update_view()
    p1.vb.sigResized.connect(update_view)  # type: ignore

    return line, p2


def get_autorange(data: ArrayLike, factor: float = 0.05):
    """Use data to determine the range for plot boundaries."""
    if np.all(np.isnan(data)):
        return 0, 1
    x1, x2 = np.nanmin(data), np.nanmax(data)
    pad = (x2 - x1) * factor
    return x1 - pad, x2 + pad

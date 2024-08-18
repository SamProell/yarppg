import pyqtgraph


def plain_image_item(data):
    """Create a `pyqtgraph.ImageView` showing only the actual image."""
    img_item = pyqtgraph.image(data)
    img_item.ui.histogram.hide()
    img_item.ui.roiBtn.hide()
    img_item.ui.menuBtn.hide()
    return img_item


def add_multiaxis_plot(p1: pyqtgraph.PlotItem, **kwargs):
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

    return line

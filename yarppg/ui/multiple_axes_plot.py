import pyqtgraph as pg


def add_plot(p1, **kwargs):
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

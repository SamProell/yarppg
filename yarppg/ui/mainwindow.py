"""Provides the MainWindow for the rPPG GUI."""
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow

from yarppg.rppg import RPPG

from . import helpers


class MainWindow(QMainWindow):
    def __init__(
        self,
        app,
        rppg: RPPG,
        winsize=(1000, 400),
        graphwin=150,
        legend=False,
        blur_roi=-1,
    ):
        QMainWindow.__init__(self)
        self._app = app

        self.rppg = rppg
        self.rppg.rppg_updated.connect(self.on_rppg_updated)
        self.rppg.new_hr.connect(self.update_hr)

        self.graphwin = graphwin

        self.lines = []
        self.plots = []
        self.auto_range_factor = 0.05

        self.init_ui(winsize=winsize)
        if legend:
            self._add_legend()
        self.blur_roi = blur_roi

    def init_ui(self, winsize):
        pg.setConfigOptions(antialias=True, foreground="k", background="w")
        self.setWindowTitle("yet another rPPG")
        self.setGeometry(0, 0, winsize[0], winsize[1])

        layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(layout)

        self.img = pg.ImageItem(axisOrder="row-major")
        vb = layout.addViewBox(
            col=0, row=0, rowspan=3, invertX=True, invertY=True, lockAspect=True
        )
        vb.addItem(self.img)

        p1 = layout.addPlot(row=0, col=1, colspan=1)
        p1.hideAxis("left")
        p1.hideAxis("bottom")
        self.lines.append(p1.plot(pen=pg.mkPen("k", width=3)))
        self.plots.append(p1)

        if self.rppg.num_processors > 1:
            p2 = layout.addPlot(row=1, col=1, colspan=1)
            p2.hideAxis("left")
            self.lines.append(p2.plot())
            self.plots.append(p2)
            for _ in range(2, self.rppg.num_processors):
                pen = pg.mkPen(width=3)
                line, plot = helpers.add_multiaxis_plot(p2, pen=pen)
                self.lines.append(line)
                self.plots.append(plot)
        for plot in self.plots:
            plot.disableAutoRange()

        self.hr_label = layout.addLabel(text="Heart rate:", row=4, col=0, size="20pt")

    @staticmethod
    def _customize_legend(line, fs="10pt", margins=(5, 0, 5, 0)):
        line.layout.setContentsMargins(*margins)
        if fs is not None:
            for _, label in line.items:
                label.setText(label.text, size=fs)

    def _add_legend(self):
        layout = self.centralWidget()
        p = layout.addPlot(row=2, col=1)
        p.hideAxis("left")
        p.hideAxis("bottom")
        legend = pg.LegendItem(verSpacing=2)
        self._customize_legend(legend)
        legend.setParentItem(p)
        for line, name in zip(self.lines, self.rppg.processor_names):
            legend.addItem(line, name)

    def update_hr(self, hr):
        self.hr_label.setText("Heart rate: {:5.1f} beat/min".format(hr))

    def on_rppg_updated(self, results):
        ts = results.ts(self.graphwin)
        for pi, vs in enumerate(results.vs_iter(self.graphwin)):
            self.lines[pi].setData(x=ts, y=vs)
            self.plots[pi].setXRange(ts[0], ts[-1])
            self.plots[pi].setYRange(*helpers.get_autorange(vs, self.auto_range_factor))

        img = results.rawimg
        roi = results.roi
        roi.pixelate_face(img, self.blur_roi)
        roi.draw_roi(img)

        self.img.setImage(img)

        print("%.3f" % results.dt, results.roi, "FPS:", int(results.fps))

    def set_pen(self, color=None, width=1, index=0):
        if index > len(self.lines):
            raise IndexError(f"index {index} too high for {len(self.lines)} lines")
        pen = pg.mkPen(color or "k", width=width)
        self.lines[index].setPen(pen)

    def execute(self):
        self.show()
        self.rppg.start()
        return self._app.exec_()

    def closeEvent(self, event):  # noqa: N802, ARG002
        self.rppg.finish()

    def keyPressEvent(self, e):  # noqa: N802
        if e.key() == ord("Q"):
            self.close()

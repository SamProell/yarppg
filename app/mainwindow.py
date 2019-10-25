import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QHBoxLayout
import pyqtgraph as pg

from .rppg import RPPG
from .utils.multiple_axes_plot import add_plot


class MainWindow(QMainWindow):
    def __init__(self, app, rppg, winsize=(1000, 400), graphwin=150,
                 legend=False):
        QMainWindow.__init__(self)
        self._app = app

        self.rppg = rppg
        self.rppg.new_update.connect(lambda code: self.updated(code))

        self.graphwin = graphwin
        self.ts = [0]

        self.img = None
        self.lines = []
        self.plots = []
        self.auto_range_factor = 0.05

        self.init_ui(winsize=winsize)
        if legend:
            self._add_legend()

    def init_ui(self, winsize):
        self.setWindowTitle("yet another rPPG")
        self.setGeometry(0, 0, winsize[0], winsize[1])

        layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(layout)
        # view = pg.GraphicsView()
        # layout = pg.GraphicsLayout()
        # view.setCentralItem(layout)
        self.img = pg.ImageItem(axisOrder="row-major")
        vb = layout.addViewBox(col=0, row=0, rowspan=3, invertX=True,
                               invertY=True, lockAspect=True)
        vb.addItem(self.img)

        p1 = layout.addPlot(row=0, col=1, colspan=1)
        p1.hideAxis("left")
        p1.hideAxis("bottom")
        self.lines.append(p1.plot(antialias=True))
        self.plots.append(p1)

        if self.rppg.num_processors > 1:
            p2 = layout.addPlot(row=1, col=1, colspan=1)
            p2.hideAxis("left")
            self.lines.append(p2.plot(antialias=True))
            self.plots.append(p2)
            for processor in range(2, self.rppg.num_processors):
                l, p = add_plot(p2, antialias=True)
                self.lines.append(l)
                self.plots.append(p)
        for p in self.plots:
            p.disableAutoRange()

    @staticmethod
    def _customize_legend(l, fs="10pt", spacing=0, margins=(5, 0, 5, 0)):
        l.layout.setSpacing(spacing)
        l.layout.setContentsMargins(*margins)
        if fs is not None:
            for _, label in l.items:
                label.setText(label.text, size=fs)

    def _add_legend(self):
        layout = self.centralWidget()
        p = layout.addPlot(row=2, col=1)
        p.hideAxis("left")
        p.hideAxis("bottom")
        leg1, leg2 = pg.LegendItem(), pg.LegendItem()
        self._customize_legend(leg1)
        self._customize_legend(leg2)
        for l, n in zip(self.lines, self.rppg.processor_names):
            leg1.addItem(l, n)
            """if l == self.lines[0]:
                leg1.addItem(l, n)
                leg1.setParentItem(self.plots[0])
            else:
                leg2.addItem(l, n)
                leg2.setParentItem(self.plots[-1])
            """
        leg1.setParentItem(p)

    def updated(self, dt):
        self.ts.append(self.ts[-1] + dt)
        img = self.rppg.output_frame

        for pi, vs in enumerate(self.rppg.get_vs(self.graphwin)):
            self.lines[pi].setData(x=self.ts[-len(vs):], y=vs)
            self.plots[pi].setXRange(self.ts[-len(vs)], self.ts[-1])
            self.plots[pi].setYRange(*self.get_range(vs))

        cv2.rectangle(img, self.rppg.roi[:2], self.rppg.roi[2:], (255, 0, 0), 3)
        self.img.setImage(img)
        print("%.3f" % dt, self.rppg.roi, "FPS:", int(self.get_fps()))

    def set_pen(self, color=None, width=1, index=0):
        if index > len(self.lines):
            raise IndexError("index={} is to high for {} lines"
                             "".format(index, len(self.lines)))
        pen = pg.mkPen(color or "w", width=width)
        self.lines[index].setPen(pen)

    def get_range(self, data):
        x1, x2 = np.min(data), np.max(data)
        pad = (x2 - x1)*self.auto_range_factor
        return x1 - pad, x2 + pad

    def get_fps(self, n=5):
        return 1/np.mean(np.diff(self.ts[-n:]))

    def execute(self):
        self.show()
        self.rppg.start()
        return self._app.exec_()

    def closeEvent(self, event):
        self.rppg.finish()

    def keyPressEvent(self, e):
        if e.key() == ord("Q"):
            self.close()

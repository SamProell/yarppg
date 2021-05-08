import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QHBoxLayout, QLabel
import pyqtgraph as pg

# from yarppg.rppg import RPPG

from .multiple_axes_plot import add_plot


class MainWindow(QMainWindow):
    def __init__(self, app, rppg, winsize=(1000, 400), graphwin=150,
                 legend=False, blur_roi=-1):
        QMainWindow.__init__(self)
        self._app = app

        self.rppg = rppg
        self.rppg.new_update.connect(self.updated)
        self.rppg.new_hr.connect(self.update_hr)

        self.graphwin = graphwin
        self.ts = [0]

        self.img = None
        self.lines = []
        self.plots = []
        self.auto_range_factor = 0.05

        self.hr_label = None

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
        vb = layout.addViewBox(col=0, row=0, rowspan=3, invertX=True,
                               invertY=True, lockAspect=True)
        vb.addItem(self.img)

        p1 = layout.addPlot(row=0, col=1, colspan=1)
        p1.hideAxis("left")
        p1.hideAxis("bottom")
        self.lines.append(p1.plot(antialias=True, pen=pg.mkPen("k", width=3)))
        self.plots.append(p1)

        if self.rppg.num_processors > 1:
            p2 = layout.addPlot(row=1, col=1, colspan=1)
            p2.hideAxis("left")
            self.lines.append(p2.plot(antialias=True))
            self.plots.append(p2)
            for processor in range(2, self.rppg.num_processors):
                l, p = add_plot(p2, antialias=True, pen=pg.mkPen(width=3))
                self.lines.append(l)
                self.plots.append(p)
        for p in self.plots:
            p.disableAutoRange()

        self.hr_label = layout.addLabel(text="Heart rate:", row=4, col=0,
                                        size="20pt")

    @staticmethod
    def _customize_legend(l, fs="10pt", margins=(5, 0, 5, 0)):
        l.layout.setContentsMargins(*margins)
        if fs is not None:
            for _, label in l.items:
                label.setText(label.text, size=fs)

    def _add_legend(self):
        layout = self.centralWidget()
        p = layout.addPlot(row=2, col=1)
        p.hideAxis("left")
        p.hideAxis("bottom")
        legend = pg.LegendItem(verSpacing=2)
        self._customize_legend(legend)
        legend.setParentItem(p)
        for l, n in zip(self.lines, self.rppg.processor_names):
            legend.addItem(l, n)

    def update_hr(self, hr):
        self.hr_label.setText("Heart rate: {:5.1f} beat/min".format(hr))

    def updated(self, dt):
        # self.ts.append(self.ts[-1] + dt)
        img = self.rppg.output_frame

        ts = self.rppg.get_ts(self.graphwin)
        for pi, vs in enumerate(self.rppg.get_vs(self.graphwin)):
            self.lines[pi].setData(x=ts, y=vs)
            self.plots[pi].setXRange(ts[0], ts[-1])
            self.plots[pi].setYRange(*self.get_range(vs))

        roi = self.rppg.roi
        cv2.rectangle(img, roi[:2], roi[2:], (255, 0, 0), 3)

        self._pixelate_roi(img, roi)
        self.img.setImage(img)
        print("%.3f" % dt, self.rppg.roi, "FPS:", int(self.rppg.get_fps()))

    def _pixelate_roi(self, img, roi):
        blursize = self.blur_roi
        if blursize > 0:
            roiw, roih = roi[2] - roi[0], roi[3] - roi[1]
            if roiw <= self.blur_roi or roih <= self.blur_roi:
                return
            slicey = slice(roi[1], roi[3])
            slicex = slice(roi[0], roi[2])
            tmp = cv2.resize(img[slicey, slicex],
                             (int(roiw/blursize), int(roih/blursize)),
                             interpolation=cv2.INTER_LINEAR)
            img[slicey, slicex] = cv2.resize(tmp, (roiw, roih),
                                             interpolation=cv2.INTER_NEAREST)

    def set_pen(self, color=None, width=1, index=0):
        if index > len(self.lines):
            raise IndexError("index={} is to high for {} lines"
                             "".format(index, len(self.lines)))
        pen = pg.mkPen(color or "k", width=width)
        self.lines[index].setPen(pen)

    def get_range(self, data):
        if np.all(np.isnan(data)):
            return 0, 1
        x1, x2 = np.nanmin(data), np.nanmax(data)
        pad = (x2 - x1)*self.auto_range_factor
        return x1 - pad, x2 + pad

    def execute(self):
        self.show()
        self.rppg.start()
        return self._app.exec_()

    def closeEvent(self, event):
        self.rppg.finish()

    def keyPressEvent(self, e):
        if e.key() == ord("Q"):
            self.close()

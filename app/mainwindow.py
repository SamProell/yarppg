import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QGridLayout
import pyqtgraph as pg

from .rppg import RPPG
from .utils.multiple_axes_plot import add_plot


class MainWindow(QMainWindow):
    def __init__(self, app, rppg, winsize=(800, 600), graphwin=150):
        QMainWindow.__init__(self)
        self._app = app

        self.rppg = rppg
        self.rppg.new_update.connect(lambda code: self.updated(code))

        self.graphwin = graphwin
        self.ts = [0]

        self.img = None
        self.lines = []

        self.init_ui(winsize=winsize)

    def init_ui(self, winsize):
        self.setWindowTitle("yet another rPPG")
        self.setGeometry(0, 0, winsize[0], winsize[1])

        layout = pg.GraphicsLayoutWidget()
        # view = pg.GraphicsView()
        # layout = pg.GraphicsLayout()
        # view.setCentralItem(layout)
        self.img = pg.ImageItem(axisOrder="row-major")
        vb = layout.addViewBox(col=0, row=0, rowspan=2, invertX=True,
                               invertY=True, lockAspect=True)
        vb.addItem(self.img)

        p1 = layout.addPlot(row=0, col=1, colspan=1)
        p1.hideAxis("left")
        p1.hideAxis("bottom")
        self.lines.append(p1.plot(antialias=True))

        if self.rppg.num_processors > 1:
            p2 = layout.addPlot(row=1, col=1, colspan=1)
            p2.hideAxis("left")
            self.lines.append(p2.plot(antialias=True))
            for processor in range(2, self.rppg.num_processors):
                self.lines.append(add_plot(p2, antialias=True))

        # self.setCentralWidget(view)
        self.setCentralWidget(layout)

    def updated(self, dt):
        self.ts.append(self.ts[-1] + dt)
        img = self.rppg.output_frame

        for pi, vs in enumerate(self.rppg.get_vs(self.graphwin)):
            self.lines[pi].setData(x=self.ts[-len(vs):], y=vs)

        cv2.rectangle(img, self.rppg.roi[:2], self.rppg.roi[2:], (255, 0, 0), 3)
        self.img.setImage(img)

        print("%.3f" % dt, self.rppg.roi, "FPS:", int(self.get_fps()), next(self.rppg.get_vs(1)))

    def set_pen(self, color=None, width=1, index=0):
        if index > len(self.lines):
            raise IndexError("index={} is to high for {} lines"
                             "".format(index, len(self.lines)))
        pen = pg.mkPen(color or "w", width=width)
        self.lines[index].setPen(pen)

    def get_fps(self, n=10):
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

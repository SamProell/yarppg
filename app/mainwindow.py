import cv2
from PyQt5.QtWidgets import QMainWindow, QGridLayout
import pyqtgraph as pg

from .rppg import RPPG


class MainWindow(QMainWindow):
    def __init__(self, app, rppg, winsize=(800, 600), graphwin=150):
        QMainWindow.__init__(self)
        self._app = app

        self.rppg = rppg
        self.rppg.new_update.connect(lambda code: self.updated(code))

        self.graphwin = graphwin
        self.ts = [0]

        self.img = None
        self.main_line = None
        self.init_ui(winsize=winsize)

    def init_ui(self, winsize):
        self.setWindowTitle("yet another rPPG")
        self.setGeometry(0, 0, winsize[0], winsize[1])

        layout = pg.GraphicsLayoutWidget()
        # view = pg.GraphicsView()
        # layout = pg.GraphicsLayout()
        # view.setCentralItem(layout)
        self.img = pg.ImageItem(axisOrder="row-major")
        vb = layout.addViewBox(col=0, row=0, rowspan=1, invertX=True,
                               invertY=True, lockAspect=True)
        vb.addItem(self.img)

        p1 = layout.addPlot(row=0, col=1, colspan=1)
        p1.hideAxis("left")
        self.main_line = p1.plot(antialias=True)

        # self.setCentralWidget(view)
        self.setCentralWidget(layout)

    def updated(self, dt):
        self.ts.append(self.ts[-1] + dt)
        img = self.rppg.output_frame

        for pi, vs in enumerate(self.rppg.get_vs(self.graphwin)):
            if pi == 0:
                self.main_line.setData(x=self.ts[-len(vs):], y=vs)

        cv2.rectangle(img, self.rppg.roi[:2], self.rppg.roi[2:], (255, 0, 0), 3)
        self.img.setImage(img)

        print(dt, self.geometry())

    def execute(self):
        self.show()
        self.rppg.start()
        return self._app.exec_()

    def closeEvent(self, event):
        self.rppg.finish()

    def keyPressEvent(self, e):
        if e.key() == ord("Q"):
            self.close()

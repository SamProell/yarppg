import cv2
from PyQt5.QtWidgets import QMainWindow, QGridLayout
import pyqtgraph as pg

from .rppg import RPPG


class MainWindow(QMainWindow):
    def __init__(self, app, rppg, winsize=(800, 600)):
        QMainWindow.__init__(self)
        self._app = app

        self.rppg = rppg
        self.rppg.new_update.connect(lambda code: self.updated(code))

        self.img = None
        self.setWindowTitle("yet another rPPG")
        self.setGeometry(0, 0, winsize[0], winsize[1])

        self.init_ui()

    def init_ui(self):
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout()
        view.setCentralItem(layout)
        self.img = pg.ImageItem(axisOrder="row-major")
        vb = layout.addViewBox(invertX=True, invertY=True, lockAspect=True)
        vb.addItem(self.img)

        self.setCentralWidget(view)

    def updated(self, dt):
        print(dt)
        img = self.rppg.output_frame
        cv2.rectangle(img, self.rppg.roi[:2], self.rppg.roi[2:], (255, 0, 0), 3)

        self.img.setImage(img)

    def execute(self):
        self.show()
        self.rppg.start()
        return self._app.exec_()

    def closeEvent(self, event):
        self.rppg.finish()

    def keyPressEvent(self, e):
        if e.key() == ord("Q"):
            self.close()

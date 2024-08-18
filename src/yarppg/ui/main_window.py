import sys
from collections import deque
from datetime import datetime

import numpy as np
import pyqtgraph
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

import yarppg
from yarppg.rppg import Rppg
from yarppg.ui import camera, utils

pyqtgraph.setConfigOptions(
    imageAxisOrder="row-major", antialias=True, foreground="k", background="w"
)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("yet another rPPG")
        self._init_ui()

        self.history = deque(maxlen=150)

    def _init_ui(self):
        child = QWidget()
        layout = pyqtgraph.QtWidgets.QHBoxLayout()
        child.setLayout(layout)
        self.setCentralWidget(child)

        self.img_item = utils.plain_image_item(np.random.randn(10, 20))
        self.img_item.setMinimumSize(640, 480)
        layout.addWidget(self.img_item)

        grid = self._make_plots()
        layout.addWidget(grid)

    def _make_plots(self) -> pyqtgraph.GraphicsLayoutWidget:
        grid = pyqtgraph.GraphicsLayoutWidget()
        self.main_plot: pyqtgraph.PlotItem = grid.addPlot(row=0, col=0)  # type: ignore
        self.rgb_plot: pyqtgraph.PlotItem = grid.addPlot(row=1, col=0)  # type: ignore

        self.main_line = self.main_plot.plot(pen=pyqtgraph.mkPen("k", width=3))
        self.rgb_lines = []
        for c in "rgb":
            pen = pyqtgraph.mkPen(c, width=1.5)
            self.rgb_lines.append(self.rgb_plot.plot(pen=pen))

        self.main_plot.hideAxis("bottom")
        self.main_plot.hideAxis("left")
        self.rgb_plot.hideAxis("left")
        return grid

    def on_frame(self, frame: np.ndarray) -> None:
        self.img_item.setImage(frame[:, ::-1])

    def on_result(self, result: yarppg.RppgResult) -> None:
        frame = result.roi.baseimg.copy()
        if result.roi.face_rect is not None:
            yarppg.pixelate(frame, result.roi.face_rect, size=10)
        self.img_item.setImage(frame)

        rgb = result.roi_mean
        self.history.append((result.value, rgb.r, rgb.g, rgb.b))

        data = np.asarray(self.history)

        self.main_line.setData(np.arange(len(data)), data[:, 0])
        for i in range(3):
            self.rgb_lines[i].setData(np.arange(len(data)), data[:, i + 1])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    cam = camera.Camera()
    cam.start()
    rppg = Rppg()

    def update(frame: np.ndarray):
        result = rppg.process_frame(frame)
        w.on_result(result)

    cam.frame_received.connect(update)
    w.show()
    app.exec()
    cam.stop()

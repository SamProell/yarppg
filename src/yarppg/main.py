"""Main entrypoint for yarppg GUI."""
import sys

import numpy as np
import pyqtgraph
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

import helpers

pyqtgraph.setConfigOptions(
    imageAxisOrder="row-major", antialias=True, foreground="k", background="w"
)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("yet another rPPG")
        self.init_ui()

    def init_ui(self):
        child = QWidget()
        layout = pyqtgraph.QtWidgets.QVBoxLayout()
        child.setLayout(layout)
        self.setCentralWidget(child)

        self.img_item = helpers.plain_image_item(np.random.randn(10, 20))
        layout.addWidget(self.img_item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()

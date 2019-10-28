import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class Camera(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, video=0, parent=None):
        QThread.__init__(self, parent=parent)
        self._cap = cv2.VideoCapture(video)
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                self._running = False
                raise RuntimeError("No frame received")
            else:
                self.new_frame.emit(frame)

    def stop(self):
        self._running = False
        time.sleep(0.1)
        self._cap.release()

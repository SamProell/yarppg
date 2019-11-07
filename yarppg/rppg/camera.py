import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class Camera(QThread):
    """Wraps an OpenCV VideoCapture and provides a Qt signal with new
    frames.

    The :py:`run` function launches a loop that waits for new frames in
    the VideoCapture and emits them with a `new_frame` signal. Calling
    :py:`stop` stops the loop and releases the camera.
    """

    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, video=0, parent=None):
        """Initialize Camera instance

        Args:
            video (int or string): ID of camera or video filename
            parent (QObject): parent object in Qt context
        """

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

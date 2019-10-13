from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject

from .camera import Camera


class RPPG(QObject):
    new_update = pyqtSignal(float)

    def __init__(self, roi_detector, smooth_roi=0, parent=None, video=0):
        QObject.__init__(self, parent)
        self.smooth_roi = smooth_roi
        self.roi = None
        self._processors = []
        self._roi_detector = roi_detector

        self._set_camera(video)

        self.last_update = datetime.now()
        self.output_frame = None

    def _set_camera(self, video):
        self._cam = Camera(video=video, parent=self)
        self._cam.new_frame.connect(lambda frame: self.frame_received(frame))

    def update_roi(self, frame):
        roi = self._roi_detector(frame)
        if self.roi is None:
            self.roi = roi
        else:
            self.roi = tuple((np.multiply(roi, 1 - self.smooth_roi)
                             + np.multiply(self.roi, self.smooth_roi)).astype(int))

    def frame_received(self, frame):
        self.output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.update_roi(frame)

        dt = (datetime.now() - self.last_update).total_seconds()
        self.last_update = datetime.now()
        self.new_update.emit(dt)

    def start(self):
        self._cam.start()

    def finish(self):
        print("finishing up...")
        self._cam.stop()

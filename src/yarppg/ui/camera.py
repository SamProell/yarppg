"""Provides compatible wrappers around the CV2 camera input."""
import time

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal


class Camera(QThread):
    """Wraps cv2.VideoCapture and emits Qt signals with frames in RGB format.

    The `run` function launches a loop that waits for new frames in
    the VideoCapture and emits them with a `new_frame` signal.
    Calling `stop` stops the loop and releases the camera.

    It is very difficult to set camera properties through OpenCV. Setting
    the `exposure` property may or may not work on your end. Range of required
    values for exposure are also badly documented and not consistent.
    See for example here:
    <https://www.principiaprogramatica.com/2017/06/11/setting-manual-exposure-in-opencv/>
    Also, settings might stay active even after closing the application.

    Args:
        video: ID of camera or video filename
        parent: parent object in Qt context
        delay_frames: delay next read until specified time passed. Defaults to NaN.
        exposure: set fixed exposure instead of auto-exposure. Defaults to None.
    """

    frame_received = pyqtSignal(np.ndarray)

    def __init__(
        self,
        video: int | str = 0,
        parent: QObject | None = None,
        delay_frames: float = np.nan,
        exposure: float | None = None,
    ):
        QThread.__init__(self, parent=parent)
        self._cap = cv2.VideoCapture(video)
        if exposure is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
            self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self._running = False
        self.delay_frames = delay_frames

    def run(self):
        """Start camera and emit successive frames."""
        self._running = True
        while self._running:
            ret, frame = self._cap.read()
            last_time = time.perf_counter()

            if not ret:
                self._running = False
                raise RuntimeError("No frame received")
            else:
                self.frame_received.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            while (time.perf_counter() - last_time) < self.delay_frames:
                # np.nan will always evaluate to False and thus skip this.
                time.sleep(0.001)

    def stop(self):
        """Stop camera loop and release resources."""
        self._running = False
        time.sleep(0.1)
        self._cap.release()

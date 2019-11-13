from datetime import datetime
import pathlib

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal, QObject

from yarppg.rppg.camera import Camera


class RPPG(QObject):
    new_update = pyqtSignal(float)
    _dummy_signal = pyqtSignal(float)

    def __init__(self, roi_detector, roi_smooth=0, parent=None, video=0,
                 hr_calculator=None):
        QObject.__init__(self, parent)
        self.roi_smooth = roi_smooth
        self.roi = None
        self._processors = []
        self._roi_detector = roi_detector

        self._set_camera(video)

        self._dts = []
        self.last_update = datetime.now()
        self.output_frame = None
        self.hr_calculator = hr_calculator

        if self.hr_calculator is not None:
            self.new_hr = self.hr_calculator.new_hr
        else:
            self.new_hr = self._dummy_signal

        self.output_filename = None

    def _set_camera(self, video):
        self._cam = Camera(video=video, parent=self)
        self._cam.new_frame.connect(lambda frame: self.frame_received(frame))

    def add_processor(self, processor):
        self._processors.append(processor)

    def update_roi(self, frame):
        roi = self._roi_detector(frame)
        if self.roi is None:
            self.roi = roi
        else:
            self.roi = tuple((np.multiply(roi, 1 - self.roi_smooth)
                              + np.multiply(self.roi, self.roi_smooth)
                              ).astype(int))

    def frame_received(self, frame):
        self.output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.update_roi(frame)

        for processor in self._processors:
            processor(frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]])
        self.hr_calculator.update(self)

        dt = (datetime.now() - self.last_update).total_seconds()
        self.last_update = datetime.now()
        self._dts.append(dt)
        self.new_update.emit(dt)

    def get_vs(self, n=None):
        for processor in self._processors:
            if n is None:
                yield np.array(processor.vs, copy=True)
            else:
                yield np.array(processor.vs[-n:], copy=True)

    def get_ts(self, n=None):
        if n is None:
            dts = self._dts
        else:
            dts = self._dts[-n:]
        return np.cumsum(dts)

    def get_fps(self, n=5):
        return 1/np.mean(self._dts[-n:])

    def save_signals(self):
        path = pathlib.Path(self.output_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = ["ts"] + ["p%d" % i for i in range(self.num_processors)]
        data = np.vstack((self.get_ts(),) + tuple(self.get_vs())).T

        df = pd.DataFrame(data=data, columns=names)
        if path.suffix == ".csv":
            df.to_csv(path, float_format="%.7f", index=False)
        elif path.suffix in {".pkl", ".pickle"}:
            df.to_pickle(path)
        elif path.suffix == ".np":
            np.save(path, data)
        elif path.suffix == ".npz":
            np.savez_compressed(path, data=data)
        else:
            raise IOError("Unknown file extension '{}'".format(path.suffix))

    @property
    def num_processors(self):
        return len(self._processors)

    @property
    def processor_names(self):
        return [str(p) for p in self._processors]

    def start(self):
        self._cam.start()

    def finish(self):
        print("finishing up...")
        if self.output_filename is not None:
            self.save_signals()
        self._cam.stop()

from collections import namedtuple
from datetime import datetime
import pathlib

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal, QObject

from yarppg.rppg.camera import Camera


def write_dataframe(path, df):
    path = pathlib.Path(path)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, float_format="%.7f", index=False)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(path)
    elif path.suffix.lower() in {".feather"}:
        df.to_feather(path)
    else:
        raise IOError("Unknown file extension '{}'".format(path.suffix))

RppgResults = namedtuple("RppgResults", ["dt",
                                         "rawimg",
                                         "roi",
                                         "hr",
                                         "vs_iter",
                                         "ts",
                                         "fps",
                                         ])


class RPPG(QObject):
    rppg_updated = pyqtSignal(RppgResults)
    _dummy_signal = pyqtSignal(float)

    def __init__(self, roi_detector, parent=None, video=0,
                 hr_calculator=None):
        QObject.__init__(self, parent)
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
        self._cam.frame_received.connect(self.on_frame_received)

    def add_processor(self, processor):
        self._processors.append(processor)

    def on_frame_received(self, frame):
        self.output_frame = frame
        self.roi = self._roi_detector(frame)

        for processor in self._processors:
            processor(self.roi)

        if self.hr_calculator is not None:
            self.hr_calculator.update(self)

        dt = self._update_time()
        self.rppg_updated.emit(RppgResults(dt=dt, rawimg=frame, roi=self.roi,
                                           hr=np.nan, vs_iter=self.get_vs,
                                           ts=self.get_ts, fps=self.get_fps()))

    def _update_time(self):
        dt = (datetime.now() - self.last_update).total_seconds()
        self.last_update = datetime.now()
        self._dts.append(dt)

        return dt

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

        df = self.get_dataframe()
        write_dataframe(path)

    def get_dataframe(self):
        names = ["ts"] + ["p%d" % i for i in range(self.num_processors)]
        data = np.vstack((self.get_ts(),) + tuple(self.get_vs())).T

        return pd.DataFrame(data=data, columns=names)

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

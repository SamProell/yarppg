"""Orchestrator class for the rPPG application."""
import pathlib
import time
from collections import namedtuple
from typing import Union

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal

from yarppg.rppg.camera import Camera


def write_dataframe(path: Union[str, pathlib.Path], df: pd.DataFrame) -> None:
    """Write data frame to disk with the format specified by file extension."""
    path = pathlib.Path(path)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, float_format="%.7f", index=False)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(path)
    elif path.suffix.lower() in {".feather"}:
        df.to_feather(path)
    else:
        raise IOError("Unknown file extension '{}'".format(path.suffix))


RppgResults = namedtuple(
    "RppgResults",
    ["dt", "rawimg", "roi", "hr", "vs_iter", "ts", "fps"],
)


class RPPG(QObject):
    rppg_updated = pyqtSignal(RppgResults)
    _dummy_signal = pyqtSignal(float)

    def __init__(self, roi_detector, parent=None, camera=None, hr_calculator=None):
        QObject.__init__(self, parent)
        self.roi = None
        self._processors = []
        self._roi_detector = roi_detector

        self._set_camera(camera)

        self._dts = []
        self.last_update = time.perf_counter()

        self.hr_calculator = hr_calculator
        if self.hr_calculator is not None:
            self.new_hr = self.hr_calculator.new_hr
        else:
            self.new_hr = self._dummy_signal

        self.output_filename = None

    def _set_camera(self, camera):
        self._cam = camera or Camera(video=0, parent=self)
        self._cam.frame_received.connect(self.on_frame_received)

    def add_processor(self, processor):
        self._processors.append(processor)

    def on_frame_received(self, frame):
        self.roi = self._roi_detector(frame)

        for processor in self._processors:
            processor(self.roi)

        if self.hr_calculator is not None:
            self.hr_calculator.update(self)

        dt = self._update_time()
        self.rppg_updated.emit(
            RppgResults(
                dt=dt,
                rawimg=frame,
                roi=self.roi,
                hr=np.nan,
                vs_iter=self.get_vs,
                ts=self.get_ts,
                fps=self.get_fps(),
            )
        )

    def _update_time(self):
        dt = time.perf_counter() - self.last_update
        self.last_update = time.perf_counter()
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
        return 1 / np.mean(self._dts[-n:])

    def save_signals(self):
        if self.output_filename is None:
            return

        path = pathlib.Path(self.output_filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        write_dataframe(path, self.get_dataframe())

    def get_dataframe(self):
        names = ["ts"] + [str(p) for p in self._processors]
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
        self.save_signals()  # save if filename was given.
        self._cam.stop()

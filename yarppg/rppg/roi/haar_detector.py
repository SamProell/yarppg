"""Face detector using OpenCV's Haar Cascade."""

import pathlib
import warnings

import cv2
import numpy as np

from . import roi_detect
from .region_of_interest import RegionOfInterest


class HaarCascadeDetector(roi_detect.ROIDetector):
    default_cascade = roi_detect.resource_path / "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        casc_file=None,
        scale_factor=1.1,
        min_neighbors=5,
        min_size=(30, 30),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = self._get_classifier(casc_file)

    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors
        )  # minSize=self.min_size)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return RegionOfInterest.from_rectangle(frame, (x, y), (x + w, y + h))

        return RegionOfInterest(frame, mask=None)

    @classmethod
    def _get_classifier(cls, casc_file: str | None):
        if casc_file is not None and pathlib.Path(casc_file).is_file():
            cascade = cv2.CascadeClassifier(casc_file)
        elif pathlib.Path(cls.default_cascade).is_file():
            warnings.warn(
                "cascade file '{}' not found, using default instead" "".format(
                    casc_file
                )
            )
            cascade = cv2.CascadeClassifier(str(cls.default_cascade))
        else:
            raise IOError("cascade file '{}' not found".format(casc_file))

        return cascade

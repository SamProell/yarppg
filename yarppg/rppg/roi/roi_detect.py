"""Region of interest (face) detectors."""
import dataclasses
import pathlib
from typing import Any, Dict

import numpy as np

from .region_of_interest import RegionOfInterest

resource_path = pathlib.Path(__file__).parent.parent / "_resources"


@dataclasses.dataclass
class ROIDetectorConfig:
    name: str
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


def get_boundingbox_from_landmarks(lms):
    """Calculate the bounding rectangle containing all landmarks."""
    xy = np.min(lms, axis=0)
    wh = np.subtract(np.max(lms, axis=0), xy)

    return np.r_[xy, wh]


class ROIDetector:
    def detect(self, frame):
        raise NotImplementedError("detect method needs to be overwritten.")

    def __call__(self, frame):
        return self.detect(frame)


class NoDetector(ROIDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        h, w = frame.shape[:2]
        return RegionOfInterest.from_rectangle(frame, (0, 0), (h, w))

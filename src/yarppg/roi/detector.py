import numpy as np

from .region_of_interest import RegionOfInterest


class ROIDetector:
    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        raise NotImplementedError("detect method needs to be overwritten.")

    def __call__(self, frame: np.ndarray) -> RegionOfInterest:
        return self.detect(frame)

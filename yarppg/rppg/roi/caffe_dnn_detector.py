"""Detect faces with OpenCV's generic Caffe DNN face detector."""

import cv2
import numpy as np

from . import roi_detect
from .region_of_interest import RegionOfInterest


class CaffeDNNFaceDetector(roi_detect.ROIDetector):
    prototxt = roi_detect.resource_path / "deploy.prototxt"
    caffemodel = (
        roi_detect.resource_path / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )

    color_mean = (128, 128, 128)

    def __init__(
        self,
        prototxt=None,
        caffemodel=None,
        blob_size=(300, 300),
        min_confidence=0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        print(self.caffemodel)
        self.blob_size = blob_size
        self.min_confidence = min_confidence
        if prototxt is None:
            prototxt = self.prototxt
        if caffemodel is None:
            caffemodel = self.caffemodel
        self.model = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.blob_size, self.color_mean)
        self.model.setInput(blob)
        detections = self.model.forward()[0, 0, ...]
        for det in detections:
            if det[2] > self.min_confidence:
                x1, y1, x2, y2 = np.multiply(det[3:7], (w, h, w, h)).astype(int)
                return RegionOfInterest.from_rectangle(frame, (x1, y1), (x2, y2))
        return RegionOfInterest(frame)

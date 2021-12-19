import os
import warnings

import cv2
import numpy as np


def exponential_smooth(new_roi, old_roi, factor):
    if factor <= 0.0 or old_roi is None:
        return new_roi

    smooth_roi = np.multiply(new_roi, 1 - factor) + np.multiply(old_roi, factor)
    return tuple(smooth_roi.astype(int))

class ROIDetector:
    def __init__(self, smooth_factor=0.0, **kwargs):
        self.oldroi = None
        self.smooth_factor = smooth_factor
        super().__init__(**kwargs)

    def detect(self, frame):
        raise NotImplementedError("detect method needs to be overwritten.")

    def get_roi(self, frame):
        roi = self.detect(frame)
        self.oldroi = exponential_smooth(roi, self.oldroi, self.smooth_factor)

        return self.oldroi

    def __call__(self, frame):
        return self.get_roi(frame)

class NoDetector(ROIDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, frame):
        h, w = frame.shape[:2]
        return 0, 0, w, h

class CaffeDNNFaceDetector(ROIDetector):
    prototxt = os.path.join(os.path.dirname(__file__),
                            "_resources/deploy.prototxt")
    caffemodel = os.path.join(os.path.dirname(__file__),
                              "_resources/res10_300x300_ssd_iter_140000_fp16"
                              ".caffemodel")
    color_mean = (128, 128, 128)

    def __init__(self, prototxt=None, caffemodel=None,
                 blob_size=(300, 300),
                 min_confidence=0.3,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.blob_size = blob_size
        self.min_confidence = min_confidence
        if prototxt is None:
            prototxt = self.prototxt
        if caffemodel is None:
            caffemodel = self.caffemodel
        self.model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.blob_size, self.color_mean)
        self.model.setInput(blob)
        detections = self.model.forward()[0, 0, ...]
        for det in detections:
            if det[2] > self.min_confidence:
                x1, y1, x2, y2 = np.multiply(
                    det[3:7], (w, h, w, h)).astype(int)
                return x1, y1, x2, y2
        return 0, 0, 0, 0


class HaarCascadeDetector(ROIDetector):
    default_cascade = "resources/haarcascade_frontalface_default.xml"

    def __init__(self,
                 casc_file,
                 scale_factor=1.1,
                 min_neighbors=5,
                 min_size=(30, 30),
                 **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = self._get_classifier(casc_file)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray,
                                              scaleFactor=self.scale_factor,
                                              minNeighbors=self.min_neighbors,
                                              )# minSize=self.min_size)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return x, y, x+w, y+h

        return 0, 0, 0, 0

    @classmethod
    def _get_classifier(cls, casc_file):
        if os.path.isfile(casc_file):
            cascade = cv2.CascadeClassifier(casc_file)
        elif os.path.isfile(cls.default_cascade):
            warnings.warn("cascade file '{}' not found, using default instead"
                          "".format(casc_file))
            cascade = cv2.CascadeClassifier(cls.default_cascade)
        else:
            raise IOError("cascade file '{}' not found".format(casc_file))

        return cascade

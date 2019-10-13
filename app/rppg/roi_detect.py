import os
import warnings

import cv2
import numpy as np


class CaffeDNNFaceDetector:
    prototxt = "resources/deploy.prototxt"
    caffemodel = "resources/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    def __init__(self, prototxt=None, caffemodel=None,
                 blob_size=(300, 300),
                 min_confidence=0.3,
                 ):
        self.blob_size = blob_size
        self.min_confidence = min_confidence
        if prototxt is None:
            prototxt = self.prototxt
        if caffemodel is None:
            caffemodel = self.caffemodel
        self.model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def detect(self, frame):
        # frame = cv2.resize(frame, self.blob_size)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.blob_size, (128, 128, 128))
        self.model.setInput(blob)
        detections = self.model.forward()[0, 0, ...]
        for det in detections:
            if det[2] > self.min_confidence:
                x1, y1, x2, y2 = np.multiply(
                    det[3:7], (w, h, w, h)).astype(int)
                return x1, y1, x2, y2
        return 0, 0, 0, 0

    def __call__(self, frame):
        return self.detect(frame)


class HaarCascadeDetector:
    default_cascade = "resources/haarcascade_frontalface_default.xml"

    def __init__(self,
                 casc_file,
                 scale_factor=1.1,
                 min_neighbors=5,
                 min_size=(30, 30),
                 ):
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

    def __call__(self, frame):
        return self.detect(frame)

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

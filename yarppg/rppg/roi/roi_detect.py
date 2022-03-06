from pathlib import Path
import warnings
import time

import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from yarppg.rppg.roi.region_of_interest import RegionOfInterest, get_default_bgmask

resource_path = Path(__file__).parent.parent / "_resources"


def exponential_smooth(new_roi, old_roi, factor):
    if factor <= 0.0 or old_roi is None:
        return new_roi

    smooth_roi = np.multiply(new_roi, 1 - factor) + np.multiply(old_roi, factor)
    return tuple(smooth_roi.astype(int))

def get_boundingbox_from_landmarks(lms):
    xy = np.min(lms, axis=0)
    wh = np.subtract(np.max(lms, axis=0), xy)

    return np.r_[xy, wh]

class ROIDetector:
    def __init__(self, smooth_factor=0.0, **kwargs):
        self.oldroi = None
        self.smooth_factor = smooth_factor
        super().__init__(**kwargs)

    def detect(self, frame):
        raise NotImplementedError("detect method needs to be overwritten.")

    def get_roi(self, frame):
        roi = self.detect(frame)
        return roi
        # self.oldroi = exponential_smooth(roi, self.oldroi, self.smooth_factor)

        # return self.oldroi

    def __call__(self, frame):
        return self.get_roi(frame)

class NoDetector(ROIDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, frame):
        h, w = frame.shape[:2]
        return RegionOfInterest.from_rectangle(frame, (0, 0), (h, w))


class CaffeDNNFaceDetector(ROIDetector):
    prototxt = resource_path / "deploy.prototxt"
    caffemodel = resource_path / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    color_mean = (128, 128, 128)

    def __init__(self, prototxt=None, caffemodel=None,
                 blob_size=(300, 300),
                 min_confidence=0.3,
                 **kwargs
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

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.blob_size, self.color_mean)
        self.model.setInput(blob)
        detections = self.model.forward()[0, 0, ...]
        for det in detections:
            if det[2] > self.min_confidence:
                x1, y1, x2, y2 = np.multiply(
                    det[3:7], (w, h, w, h)).astype(int)
                return RegionOfInterest.from_rectangle(frame, (x1, y1), (x2, y2))
        return RegionOfInterest(frame)


class HaarCascadeDetector(ROIDetector):
    default_cascade = resource_path / "haarcascade_frontalface_default.xml"

    def __init__(self,
                 casc_file=None,
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
            return RegionOfInterest.from_rectangle(frame, (x, y), (x+w, y+h))

        return RegionOfInterest(frame, mask=None)

    @classmethod
    def _get_classifier(cls, casc_file: str):
        if casc_file is not None and Path(casc_file).is_file():
            cascade = cv2.CascadeClassifier(casc_file)
        elif Path(cls.default_cascade).is_file():
            warnings.warn("cascade file '{}' not found, using default instead"
                          "".format(casc_file))
            cascade = cv2.CascadeClassifier(str(cls.default_cascade))
        else:
            raise IOError("cascade file '{}' not found".format(casc_file))

        return cascade


def get_facemesh_coords(landmark_list, frame):
    h, w = frame.shape[:2]
    xys = [(landmark.x, landmark.y) for landmark in landmark_list.landmark]

    return np.multiply(xys, [w, h]).astype(int)

class FaceMeshDetector(ROIDetector):
    _lower_face = [200, 431, 411, 340, 349, 120, 111, 187, 211]

    def __init__(self, draw_landmarks=False, refine=False, **kwargs):
        super().__init__(**kwargs)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw_landmarks=draw_landmarks

    def __del__(self):
        self.face_mesh.close()

    def detect(self, frame):
        rawimg = frame.copy()

        frame.flags.writeable = False
        results = self.face_mesh.process(frame)
        frame.flags.writeable = True

        if results.multi_face_landmarks is None:
            return RegionOfInterest(frame, mask=None)

        if self.draw_landmarks:
            self.draw_facemesh(frame, results.multi_face_landmarks,
                               tesselate=True)

        landmarks = get_facemesh_coords(results.multi_face_landmarks[0], frame)
        facerect = get_boundingbox_from_landmarks(landmarks)
        bgmask = get_default_bgmask(frame.shape[1], frame.shape[0])

        return RegionOfInterest.from_contour(rawimg, landmarks[self._lower_face],
                                             facerect=facerect, bgmask=bgmask)

    def draw_facemesh(self, img, multi_face_landmarks, tesselate=False,
                      contour=False, irises=False):
        if multi_face_landmarks is None:
            return

        for face_landmarks in multi_face_landmarks:
            if tesselate:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            if contour:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
            if irises and len(face_landmarks) > 468:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

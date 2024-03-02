"""Detect the lower face with MediaPipe's FaceMesh detector."""

import mediapipe as mp
import numpy as np

from . import region_of_interest, roi_detect


def get_facemesh_coords(landmark_list, frame):
    """Get unnormalized coordinates of face mesh landmarks."""
    h, w = frame.shape[:2]
    xys = [(landmark.x, landmark.y) for landmark in landmark_list.landmark]

    return np.multiply(xys, [w, h]).astype(int)


class FaceMeshDetector(roi_detect.ROIDetector):
    _lower_face = [200, 431, 411, 340, 349, 120, 111, 187, 211]

    def __init__(self, draw_landmarks=False, refine=False, **kwargs):
        super().__init__(**kwargs)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.draw_landmarks = draw_landmarks

    def __del__(self):
        self.face_mesh.close()

    def detect(self, frame: np.ndarray) -> region_of_interest.RegionOfInterest:
        rawimg = frame.copy()

        frame.flags.writeable = False
        results = self.face_mesh.process(frame)
        frame.flags.writeable = True

        if results.multi_face_landmarks is None:
            return region_of_interest.RegionOfInterest(frame, mask=None)

        if self.draw_landmarks:
            self.draw_facemesh(frame, results.multi_face_landmarks, tesselate=True)

        landmarks = get_facemesh_coords(results.multi_face_landmarks[0], frame)
        facerect = roi_detect.get_boundingbox_from_landmarks(landmarks)
        bgmask = region_of_interest.get_default_bgmask(frame.shape[1], frame.shape[0])

        return region_of_interest.RegionOfInterest.from_contour(
            rawimg, landmarks[self._lower_face], facerect=facerect, bgmask=bgmask
        )

    def draw_facemesh(
        self,
        img,
        multi_face_landmarks,
        tesselate=False,
        contour=False,
        irises=False,
    ):
        if multi_face_landmarks is None:
            return

        for face_landmarks in multi_face_landmarks:
            if tesselate:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                )
            if contour:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                )
            if irises and len(face_landmarks) > 468:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

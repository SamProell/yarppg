"""Detect the lower face with MediaPipe's FaceMesh detector."""

import pathlib
import time
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from . import region_of_interest, roi_detect

MEDIAPIPE_MODELS_BASE = "https://storage.googleapis.com/mediapipe-models/"
LANDMARKER_TASK = "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

TESSELATION_SPEC = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()  # type: ignore
CONTOUR_SPEC = mp.solutions.drawing_styles.get_default_face_mesh_contours_style()  # type: ignore
IRISES_SPEC = mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()  # type: ignore


def _download_cached_file(local_file, remote_file):
    try:
        if not pathlib.Path(local_file).exists():
            urllib.request.urlretrieve(remote_file, filename=str(local_file))
            assert pathlib.Path(local_file).exists()
    except:  # noqa: E722
        return None
    return local_file


def get_face_landmarker_modelfile():
    """Get the filename of the FaceLandmarker - download file if necessary."""
    task_filename = roi_detect.resource_path / "face_landmarker.task"
    return _download_cached_file(task_filename, MEDIAPIPE_MODELS_BASE + LANDMARKER_TASK)


def get_landmark_coords(
    landmarks: list[NormalizedLandmark], width: int, height: int
) -> np.ndarray:
    """Extract normalized landmark coordinates to array of pixel coordinates."""
    xyz = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return np.multiply(xyz, [width, height, width]).astype(int)


class FaceMeshDetector(roi_detect.ROIDetector):
    _lower_face = [200, 431, 411, 340, 349, 120, 111, 187, 211]

    def __init__(self, draw_landmarks=False, **kwargs):
        super().__init__(**kwargs)
        modelpath = get_face_landmarker_modelfile()
        if modelpath is None:
            raise FileNotFoundError("Could not find or download landmarker model file.")
        base_options = mp.tasks.BaseOptions(model_asset_path=modelpath)
        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_options
        )
        self.draw_landmarks = draw_landmarks

    def __del__(self):
        self.landmarker.close()

    def detect(self, frame: np.ndarray) -> region_of_interest.RegionOfInterest:
        rawimg = frame.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self.landmarker.detect_for_video(
            mp_image, int(time.perf_counter() * 1000)
        )

        if len(results.face_landmarks) < 1:
            return region_of_interest.RegionOfInterest(frame, mask=None)

        if self.draw_landmarks:
            self.draw_facemesh(frame, results.face_landmarks[0], tesselate=True)

        height, width = frame.shape[:2]
        landmarks = get_landmark_coords(results.face_landmarks[0], width, height)[:, :2]
        facerect = roi_detect.get_boundingbox_from_landmarks(landmarks)
        bgmask = region_of_interest.get_default_bgmask(frame.shape[1], frame.shape[0])

        return region_of_interest.RegionOfInterest.from_contour(
            rawimg, landmarks[self._lower_face], facerect=facerect, bgmask=bgmask
        )

    def draw_facemesh(
        self,
        img,
        face_landmarks,
        tesselate=False,
        contour=False,
        irises=False,
    ):
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(  # type: ignore
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )
        if tesselate:
            mp.solutions.drawing_utils.draw_landmarks(  # type: ignore
                image=img,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,  # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=TESSELATION_SPEC,
            )
        if contour:
            mp.solutions.drawing_utils.draw_landmarks(  # type: ignore
                image=img,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,  # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=CONTOUR_SPEC,
            )
        if irises:
            mp.solutions.drawing_utils.draw_landmarks(  # type: ignore
                image=img,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,  # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=IRISES_SPEC,
            )

"""Detect the lower face with MediaPipe's FaceMesh detector."""

import time

import mediapipe as mp
import numpy as np

from ..helpers import get_cached_resource_path
from .detector import ROIDetector
from .region_of_interest import RegionOfInterest

MEDIAPIPE_MODELS_BASE = "https://storage.googleapis.com/mediapipe-models/"
SELFIE_TASK = "image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"  # noqa: E501


def get_selfie_segmetner_modelfile():
    """Get the filename of the FaceLandmarker - download file if necessary."""
    task_filename = "selfie_multiclass.tflite"
    return get_cached_resource_path(task_filename, MEDIAPIPE_MODELS_BASE + SELFIE_TASK)


class SelfieDetector(ROIDetector):
    def __init__(self, confidence=0.5, **kwargs):
        super().__init__(**kwargs)
        self.confidence = confidence

        modelpath = get_selfie_segmetner_modelfile()
        if modelpath is None:
            raise FileNotFoundError("Could not find or download landmarker model file.")

        base_options = mp.tasks.BaseOptions(model_asset_path=modelpath)
        segmenter_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options, running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
            segmenter_options
        )

    def __del__(self):
        self.segmenter.close()

    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        rawimg = frame.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        results = self.segmenter.segment_for_video(
            mp_image, int(time.perf_counter() * 1000)
        )

        face_mask = results.confidence_masks[3].numpy_view() > self.confidence
        bg_mask = results.confidence_masks[0].numpy_view() > self.confidence
        return RegionOfInterest(
            face_mask.astype(np.uint8), baseimg=rawimg, bg_mask=bg_mask.astype(np.uint8)
        )

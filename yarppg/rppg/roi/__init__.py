"""Region of interest detection and manipulation."""

from . import caffe_dnn_detector, facemesh_detector, haar_detector
from .region_of_interest import RegionOfInterest
from .roi_detect import NoDetector, ROIDetector, ROIDetectorConfig


def get_roi_detector(cfg: ROIDetectorConfig) -> ROIDetector:
    """Initialize the ROI detector based on given config."""
    if cfg.name == "full":
        return NoDetector(**cfg.kwargs)
    elif cfg.name == "facemesh":
        return facemesh_detector.FaceMeshDetector(**cfg.kwargs)
    elif cfg.name == "caffednn":
        return caffe_dnn_detector.CaffeDNNFaceDetector(**cfg.kwargs)
    elif cfg.name == "haar":
        return haar_detector.HaarCascadeDetector(**cfg.kwargs)

    raise NotImplementedError("Config not understood: %s", cfg)

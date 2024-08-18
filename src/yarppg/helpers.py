"""Utility functions and helpers."""

import pathlib
import urllib.request
from typing import Iterator

import cv2
import numpy as np
import pyqtgraph

RESOURCE_DIR = pathlib.Path(__file__).parent / "_resources"


def plain_image_item(data):
    """Create a `pyqtgraph.ImageView` showing only the actual image."""
    img_item = pyqtgraph.image(data)
    img_item.ui.histogram.hide()
    img_item.ui.roiBtn.hide()
    img_item.ui.menuBtn.hide()
    return img_item


def get_cached_resource_path(filename: str, url: str, reload: bool = False):
    """Download a file from the web and store it locally."""
    RESOURCE_DIR.mkdir(exist_ok=True)
    local_file = RESOURCE_DIR / filename
    if not local_file.exists() or reload:
        urllib.request.urlretrieve(url, filename=str(local_file))
        if not local_file.exists():
            raise FileNotFoundError(
                f"Something went wrong when getting {filename=:!r} from {url=:!r}."
            )
    return local_file


def frames_from_video(filename: str) -> Iterator[np.ndarray]:
    """Read and yield frames from a video file."""
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def get_video_fps(filename: str) -> float:
    """Find the frame rate of the given video file."""
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

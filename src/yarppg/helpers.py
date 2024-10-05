"""Utility functions and helpers."""

import pathlib
import urllib.request
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import ArrayLike

RESOURCE_DIR = pathlib.Path(__file__).parent / "_resources"


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


def frames_from_video(filename: str | pathlib.Path) -> Iterator[np.ndarray]:
    """Read and yield frames from a video file."""
    cap = cv2.VideoCapture(str(filename))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def get_video_fps(filename: str | pathlib.Path) -> float:
    """Find the frame rate of the given video file."""
    cap = cv2.VideoCapture(str(filename))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def bpm_from_frames_per_beat(hr: ArrayLike, fps: float) -> np.ndarray:
    """Convert frames per beat to beats per minute (60 * fps / hr)."""
    return 60 * fps / np.asarray(hr)

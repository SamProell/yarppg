"""Provides the base container for regions of interests."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RegionOfInterest:
    """Container for defining the region of interest (and background) in an image."""

    mask: np.ndarray
    baseimg: np.ndarray
    bg_mask: np.ndarray | None = None
    face_rect: tuple[int, int, int, int] | None = None


def pixelate(img: np.ndarray, xywh: tuple[int, int, int, int], size: int):
    """Blur a rectangular region with oversized pixels."""
    x, y, w, h = xywh
    slicex = slice(x, x + w)
    slicey = slice(y, y + h)

    tmp = cv2.resize(
        img[slicey, slicex],
        (w // size, h // size),
        interpolation=cv2.INTER_LINEAR,
    )
    img[slicey, slicex] = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelate_mask(img: np.ndarray, mask: np.ndarray, size: int = 10):
    """Blur the bounding box of a mask with oversized pixels."""
    bbox: tuple[int, int, int, int] = cv2.boundingRect(mask)  # type: ignore
    pixelate(img, bbox, size=size)

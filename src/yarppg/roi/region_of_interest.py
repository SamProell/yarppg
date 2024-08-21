"""Provides the base container for regions of interests."""
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike


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


def contour_to_mask(size: tuple[int, int], points: ArrayLike) -> np.ndarray:
    """Create a binary mask filled inside the polygon defined by the given points.

    Args:
        size: height and width of the target image.
        points: list of polygon coordinates.

    Returns:
        A binary mask of the desired size, filled with ones.
    """
    mask = np.zeros(size, dtype="uint8")
    contours = np.reshape(np.asarray(points), (1, -1, 1, 2))
    return cv2.drawContours(mask, contours, 0, color=1, thickness=cv2.FILLED)  # type: ignore


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay masked region in an image with a transparent color.

    Args:
        img: OpenCV-compatible image.
        mask: boolean mask defining the pixels to be overlayed.
        color: base color of the overlay. Defaults to red.
        alpha: intensity of the overlay. 0 (empty) to 1 (solid). Defaults to 0.5.

    Returns:
        OpenCV image.
    """
    overlay = img.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

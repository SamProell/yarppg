import functools

from multiprocessing.sharedctypes import Value
import cv2
import numpy as np


def pixelate(img, xywh, blur):
    if blur > 0:
        x, y, w, h = xywh
        slicex = slice(x, x+w)
        slicey = slice(y, y+h)

        tmp = cv2.resize(img[slicey, slicex], (w//blur, h//blur),
                         interpolation=cv2.INTER_LINEAR)
        img[slicey, slicex] = cv2.resize(tmp, (w, h),
                                         interpolation=cv2.INTER_NEAREST)

@functools.lru_cache(maxsize=2)
def get_default_bgmask(w, h):
    mask = np.zeros((h, w), dtype="uint8")
    cv2.rectangle(mask, (0, 0), (w, 5), 255, -1)

    return mask

class RegionOfInterest:
    def __init__(self, base_img, mask=None, bgmask=None, facerect=None):
        self.rawimg = base_img

        self._mask = mask
        self._rectangle = None
        self._empty = True
        self._rectangular = False
        self._contours = None
        self._bgmask = bgmask
        self._facerect = facerect

        if mask is not None:
            self._rectangle = cv2.boundingRect(mask)
            self._empty = (self._rectangle[2] == 0 or self._rectangle[3] == 0)

    @classmethod
    def from_rectangle(cls, base_img, p1, p2, **kwargs):
        # https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        cv2.rectangle(mask, p1, p2, 255, cv2.FILLED)

        roi = RegionOfInterest(base_img, mask=mask, **kwargs)
        roi._rectangular = True

        return roi

    @classmethod
    def from_contour(cls, base_img, pointlist, **kwargs):
        # pointlist with shape nx2
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        contours = np.reshape(pointlist, (1, -1, 1, 2))
        cv2.drawContours(mask, contours, 0, color=255, thickness=cv2.FILLED)

        roi = RegionOfInterest(base_img, mask, **kwargs)
        roi._contours = contours

        return roi

    def draw_roi(self, img, color=(255, 0, 0), thickness=3):
        if self.is_empty():
            return

        if self.is_rectangular():
            p1, p2 = self.get_bounding_box(as_corners=True)
            cv2.rectangle(img, p1, p2, color, thickness)
        else:
            cv2.drawContours(img, self._contours, 0, color=color,
                             thickness=thickness)

    def pixelate_face(self, img, blursize):
        if not self.is_empty():
            xywh = self._rectangle if self._facerect is None else self._facerect
            pixelate(img, xywh, blursize)

    def is_rectangular(self):
        return self._rectangular

    def is_empty(self):
        return self._empty

    def get_bounding_box(self, as_corners=False):
        """Bounding box specified as (x, y, w, h) or min/max corners
        """
        if as_corners:
            x, y, w, h = self._rectangle
            return (x, y), (x+w, y+h)
        return self._rectangle

    def get_mean_rgb(self, background=False):
        mask = self._mask
        if background:
            if self._bgmask is None:
                raise ValueError("Background mask is not specified")
            mask = self._bgmask

        r, g, b, a = cv2.mean(self.rawimg, mask)
        return r, g, b

    def __str__(self):
        if self.is_empty():
            return "RegionOfInterest(empty)"
        if self.is_rectangular():
            return f"RegionOfInterest(rect={self._rectangle})"

        return f"RegionOfInterest(masked within bb={self._rectangle})"

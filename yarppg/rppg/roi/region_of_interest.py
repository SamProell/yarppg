import cv2
import numpy as np


class RegionOfInterest:
    def __init__(self, base_img, mask=None):
        self.rawimg = base_img

        self._mask = mask
        self._empty = False
        if mask is None:
            self._empty = True

        self._rectangular = False
        self._rectangle = None
        self._contours = None
        # TODO: Calc bounding box for non-rectangular masks

    @classmethod
    def from_rectangle(cls, base_img, p1, p2):
        # https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        cv2.rectangle(mask, p1, p2, 255, cv2.FILLED)
        roi = RegionOfInterest(base_img, mask=mask)
        roi._rectangular = True
        x, y, w, h = cv2.boundingRect(mask)
        roi._rectangle = x, y, w, h
        if (w+h) > 0:
            roi._empty = False

        return roi

    @classmethod
    def from_contour(cls, base_img, pointlist):
        # pointlist with shape nx2
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        contours = np.reshape(pointlist, (1, -1, 1, 2))
        cv2.drawContours(mask, contours, 0, color=255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(mask)

        roi = RegionOfInterest(base_img, mask)
        roi._rectangle = x, y, w, h
        roi._contours = contours
        if (w+h) == 0:
            roi._empty = True

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

    def get_mean_rgb(self):
        r, g, b, a = cv2.mean(self.rawimg, self._mask)
        return r, g, b

    def __str__(self):
        if self.is_empty():
            return "RegionOfInterest(empty)"
        if self.is_rectangular():
            return f"RegionOfInterest(rect={self._rectangle})"

        return f"RegionOfInterest(masked within bb={self._rectangle})"

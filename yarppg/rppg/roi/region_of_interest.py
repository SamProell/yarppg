import cv2


class RegionOfInterest:
    def __init__(self, mask=None):
        if mask is None:
            self._empty = True
        self._empty = False

        self._rectangular = False
        self._rectangle = None
        self._mask = mask

    @classmethod
    def from_rectangle(cls, x, y, w, h):
        roi = RegionOfInterest(mask=None)
        roi._rectangular = True
        roi._rectangle = (x, y, w, h)
        if (w+h) > 0:
            roi._empty = False

    def draw_roi(self, img, color=(255, 0, 0), thickness=3):
        if self.is_empty:
            return

        if self.is_rectangular:
            cv2.rectangle(img, self._rectangle[:2], self._rectangle[2:], color,
                          thickness)

    @property
    def is_rectangular(self):
        return self._rectangular

    @property
    def is_empty(self):
        return self._empty

    @property
    def xywh(self):
        return self._rectangle

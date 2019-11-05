
from .processor import Processor


class ColorMeanProcessor(Processor):
    channel_dict = dict(r=0, g=1, b=2)
    channel_dict_reverse = {0: "r", 1: "g", 2: "b"}

    def __init__(self, channel="g", winsize=1):
        Processor.__init__(self)

        if channel not in self.channel_dict.keys():
            raise KeyError("channel has to be one of "
                           "{}".format(set(self.channel_dict.keys())))

        self.channel = self.channel_dict[channel]
        self.winsize = winsize
        self._tmp = []

    def calculate(self, roi_pixels):
        rgb = self.spatial_pooling(roi_pixels, append_rgb=False)
        self._tmp.append(rgb[self.channel])
        return self.moving_average_update(0, self._tmp, self.winsize)

    def __str__(self):
        if self.name is None:
            channel = self.channel_dict_reverse[self.channel]
            return "ColorMean(winsize={},c={})".format(self.winsize, channel)
        return self.name

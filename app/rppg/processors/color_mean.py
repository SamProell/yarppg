
from .processor import Processor


class ColorMean(Processor):
    channel_dict = dict(r=0, g=1, b=2)

    def __init__(self, channel="g"):
        Processor.__init__(self)

        if channel not in self.channel_dict.keys():
            raise KeyError("channel has to be one of "
                           "{}".format(set(self.channel_dict.keys())))

        self.channel = self.channel_dict[channel]

    def calculate(self, roi):
        rgb = self.spatial_pooling(roi, append_rgb=False)
        self.vs.append(rgb[self.channel])

        return self.vs[-1]

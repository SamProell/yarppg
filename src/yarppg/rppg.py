import numpy as np

from . import helpers, processors, roi


class Rppg:
    def __init__(self, roi_detector: roi.ROIDetector, processor: processors.Processor):
        self.roi_detector = roi_detector
        self.processor = processor
        self.history: list[processors.RppgResult] = []

    def process_frame(self, frame: np.ndarray):
        roi = self.roi_detector.detect(frame)
        result = self.processor.process(frame, roi)
        # result.hr = self.hr_calculator.update(result)

        self.history.append(result)
        return result

    def process_video(self, filename: str) -> list[processors.RppgResult]:
        results = []
        fps = helpers.get_video_fps(filename)
        for frame in helpers.frames_from_video(filename):
            results.append(self.process_frame(frame))
        return results

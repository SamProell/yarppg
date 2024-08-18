# ruff: noqa
# %%
#! %load_ext autoreload
#! %autoreload 2
# %%
import pathlib

#! %cd {pathlib.Path(__file__).parent.parent}
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.yarppg.roi.facemesh_segmenter import FaceMeshDetector
from src.yarppg.roi.selfie_segmenter import SelfieDetector

# %%
frame = cv2.imread("tests/face_example1.jpg")
detector = FaceMeshDetector()
roi = detector.detect(frame)
# %%
plt.imshow(roi.mask)
# %%
detector = SelfieDetector()
roi = detector.detect(frame)
plt.imshow(roi.mask)
# %%
import src.yarppg.helpers
from src.yarppg.processors.processor import Processor, RppgResult
from src.yarppg.processors.chrom import ChromProcessor

from yarppg.rppg.roi.facemesh_detector import FaceMeshDetector as OldFaceMeshDetector
from yarppg.rppg.processors.chrom import ChromProcessor as OldChromProcessor
from yarppg.rppg.roi import RegionOfInterest as OldRegionOfInterest

old_detector = OldFaceMeshDetector()
old_chrom = OldChromProcessor()

# processor = Processor()
processor = ChromProcessor()
detector = FaceMeshDetector()
results: list[RppgResult] = []
old_results: list[float] = []

for frame in src.yarppg.helpers.frames_from_video("video.mp4"):
    roi = detector.detect(frame)
    results.append(processor.process(frame, roi))
    # print(results[-1].value)
    # old_roi = old_detector.detect(frame)
    old_results.append(old_chrom.calculate(OldRegionOfInterest(frame, roi.mask)))
plt.plot([r.value for r in results])
plt.plot(old_results)
# %%

import scipy.signal

from src.yarppg.rppg import Rppg
from src.yarppg.digital_filter import DigitalFilter
from src.yarppg.hr_calculator import PeakBasedHrCalculator
from src.yarppg.processors import FilteredProcessor

fs = src.yarppg.helpers.get_video_fps("video.mp4")

b, a = scipy.signal.iirfilter(2, [1.5], fs=fs, btype="low")
livefilter = DigitalFilter(b, a, xi=-1)

hrcalc = PeakBasedHrCalculator(fs, window_seconds=5)

rppg = Rppg(detector, FilteredProcessor(processor, livefilter), hrcalc)

results = rppg.process_video("video.mp4")
yfilt = np.array([r.value for r in results])
# %%

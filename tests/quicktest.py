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
import yarppg

# %%
frame = cv2.imread("tests/face_example1.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
detector = yarppg.FaceMeshDetector()
roi = detector.detect(frame)
# %%
plt.imshow(yarppg.roi.overlay_mask(roi.baseimg, roi.mask == 1, alpha=0.3))
# %%
detector = yarppg.SelfieDetector()
roi = detector.detect(frame)
plt.imshow(yarppg.roi.overlay_mask(roi.baseimg, roi.mask == 1, alpha=0.3))
# %%
from yarppg_old.rppg.roi.facemesh_detector import (
    FaceMeshDetector as OldFaceMeshDetector,
)
from yarppg_old.rppg.processors.chrom import ChromProcessor as OldChromProcessor
from yarppg_old.rppg.roi import RegionOfInterest as OldRegionOfInterest

old_detector = OldFaceMeshDetector()
old_chrom = OldChromProcessor()

# processor = yarppg.Processor()
processor = yarppg.ChromProcessor()
detector = yarppg.FaceMeshDetector()
results: list[yarppg.RppgResult] = []
old_results: list[float] = []

for frame in yarppg.frames_from_video("video.mp4"):
    roi = detector.detect(frame)
    results.append(processor.process(roi))
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

fs = yarppg.get_video_fps("video2.mp4")

b, a = scipy.signal.iirfilter(2, [0.5, 2], fs=fs, btype="bandpass")
livefilter = DigitalFilter(b, a, xi=-1)

hrcalc = PeakBasedHrCalculator(fs, window_seconds=5)

rppg = Rppg(detector, FilteredProcessor(yarppg.Processor(), livefilter), hrcalc)

results = rppg.process_video("video2.mp4")
yfilt = np.array([r.value for r in results])
plt.plot(yfilt[90:])
# %%
from yarppg import settings
import omegaconf

cfg = omegaconf.OmegaConf.structured(settings.Settings)
settings.flatten_dict(omegaconf.OmegaConf.to_container(cfg))  # type: ignore

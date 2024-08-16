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
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = detector.detect(frame)
    results.append(processor.process(frame, roi))
    # print(results[-1].value)
    # old_roi = old_detector.detect(frame)
    old_results.append(old_chrom.calculate(OldRegionOfInterest(frame, roi.mask)))
plt.plot([r.value for r in results])
plt.plot(old_results)
# %%

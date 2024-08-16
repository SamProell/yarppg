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
#! %%prun
from src.yarppg.processors.processor import Processor, RppgResult

processor = Processor()
detector = FaceMeshDetector()
results: list[RppgResult] = []
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = detector.detect(frame)
    results.append(processor.process(frame, roi))
    print(results[-1].value)
plt.plot([r.value for r in results])
# %%

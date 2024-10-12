"""A walk through the inner workings of yarPPG."""
# %% [markdown]
# # Fine-grained control of the rPPG components
# This guide walks you through the (very simple) inner workings of the
# [`Rppg.process_frame`][yarppg.rppg.Rppg.process_frame] method.
# %%
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

import yarppg

filename = "../tests/testvideo_30fps.mp4"
fps = yarppg.helpers.get_video_fps(filename)
# %% [markdown]
# ## Overview
# Remote Photoplethysmography (rPPG) typically involves three steps:
#
# - region of interest (ROI) identification
# - signal extraction
# - heart rate estimation
#
# The yarppg.Rppg class combines all these steps in a convenient manner.
# By default, `yarppg.Rppg()` will give you a simple processor, that
# finds the lower part of a face and extracts the average green channel
# within the region.
# %%
rppg = yarppg.Rppg()
# %% [markdown]
# If you are only interested in the results, you can do something like
# this:
# %%
results: list[yarppg.RppgResult] = []
for frame in yarppg.frames_from_video(filename):
    results.append(rppg.process_frame(frame))
plt.plot(np.array(results)[:, 0])
# %% [markdown]
# Under the hood, `rppg` calls
#
# 1. a roi detector's `detect` method
# 2. a signal extracor's (`yarppg.Processor`) `process`
# 3. a `HrCalculator`s `update` method.
#
# Let's define each component separately.
#
#
# ## Region of interest detection
# yarPPG comes with several different implementations of ROI detectors.
# By default, we use an AI-based face landmarker provided through Google's
# [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python).
# The `FaceMeshDetector` applies the face landmarker model and extracts the region
# of the lower face, as is done by Li et al. (2014).
#
# > X. Li, J. Chen, G. Zhao, and M. Pietikainen, “Remote Heart Rate Measurement
# From Face Videos Under Realistic Situations”, Proceedings of the IEEE Conference
# on Computer Vision and Pattern Recognition (CVPR), pp. 4264-4271, 2014
# [doi:10.1109/CVPR.2014.543](https://doi.org/10.1109/CVPR.2014.543)
# %%
roi_detector = yarppg.FaceMeshDetector()

frame = next(yarppg.frames_from_video(filename))
roi = roi_detector.detect(frame)
plt.imshow(roi.mask > 0, cmap="Greys_r", aspect="auto")

assert roi.face_rect is not None  # FaceMeshDetector also provides a bounding box.
x, y, w, h = roi.face_rect
rect = matplotlib.patches.Rectangle((x, y), w, h, edgecolor="r", facecolor="none")
plt.gca().add_patch(rect)
# %% [markdown]
# ## Signal extraction
# The default signal extractor (`yarppg.Processor`) simply calculates the average
# green channel within the region of interest. This can already be enough to
# estimate heart rate accurately, if there is no movement and lighting changes.
#
# The processor returns an `RppgResult` container, which includes some additional
# information besides the extracted value.
# For example, all processors return the mean color (R, G and B) of the ROI,
# regardless of the specific algorithm.
# %%
processor = yarppg.Processor()
result = processor.process(roi)
print(result.value == result.roi_mean.g)
# %% [markdown]
# ## Heart rate estimation
# In order to perform heart rate estimation, we need to look at the rPPG signal
# over time. The `yarppg.HrCalculator` keeps an internal buffer of recent signal
# values and periodically updates the heart rate estimate.
# The `PeakBasedHrCalculator` identifies peaks in the rPPG signal and calculates
# heart rate from the average distance of peaks within the buffer window.
#
# In your signal processing loop, you can call the `update` method in every iteration.
# The calculator will decide based on the `update_interval` attribute, whether to
# perform the calculation or to simply store the value in the buffer.
# Below, we set up the calculator to produce a new HR estimate with every 15th
# frame.
# %%
hrcalc = yarppg.PeakBasedHrCalculator(
    fs=30, window_seconds=5, distance=0.6, update_interval=15
)
# %% [markdown]
# We can see the internal buffer growing. Note that HR will be nan as long as the
# window size is not reached.
# To clear the buffer, we call hrcalc.reset()
# %%
for res in results[:5]:
    hr = hrcalc.update(res.value)
    print(len(hrcalc.values), hr)
hrcalc.reset()
print(len(hrcalc.values), "state cleared.")
# %%

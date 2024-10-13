# %% [markdown]
# yarPPG can now also be applied fully offline, without a user interface.
# To streamline such use cases, the `Rppg` orchestrator provides a helper
# function to process a video file in one line:
# [`process_video`](/reference/rppg#yarppg.rppg.Rppg.process_video).
# %%
import matplotlib.pyplot as plt
import numpy as np

import yarppg

# %% [markdown]
# ## Setup
# For this demo, we set up a bandpass-filtered version of the default processor,
# which extracts the average green channel of the region of interest.
# We need to know the number of frames per second (FPS) of the video file, to
# properly set up the filter.
# [`yarppg.get_video_fps`](/reference/helpers#yarppg.helpers.frames_from_video)
# uses OpenCV to extract the FPS information from the video file.
# %%
filename = "tests/testvideo_30fps.mp4"

fps = yarppg.get_video_fps(filename)
filter_cfg = yarppg.digital_filter.FilterConfig(fps, 0.5, 1.5, btype="bandpass")
livefilter = yarppg.digital_filter.make_digital_filter(filter_cfg)
processor = yarppg.FilteredProcessor(yarppg.Processor(), livefilter=livefilter)
# %% [markdown]
# Since the example video is quite short, we modify the behavior of the
# heart rate calculator, to produce an update with a window length of only
# four seconds.
# In practice, this will result in less accurate estimations, as outliers
# contribute more to the final result.
# Additionally, since the processors need a few seconds to adjust to the
# specific video content (lighting, colors, etc.), the provided estimates
# for this 10s video are of low quality.
# %%
rppg = yarppg.Rppg(
    processor=processor,
    hr_calc=yarppg.PeakBasedHrCalculator(fps, window_seconds=4),
)
# %% [markdown]
# ## Processing the video
# Once setup, we can process the video in just one line. By default,
# `process_video` returns a list of `RppgResult` containers. Beware
# that these include the raw image data from all frames and ROI masks
# inside the [`RegionOfInterest`
# container](/reference/containers#yarppg.containers.RegionOfInterst).
# %%
results = rppg.process_video(filename)
# %% [markdown]
# ## Handling results
# `RppgResult` allows easy conversion to an array. Even the list of results can
# be converted neatly. This produces an Nx8 array with the following values for
# each frame:
# ```
# value, roi_r, roi_g, roi_b, bg_r, bg_g, bg_b, hr
# ```
# .
# %%
values = np.array(results)
hrs = yarppg.bpm_from_frames_per_beat(values[:, -1], fps)
plt.plot(values[int(2.5 * fps) :, 0])
plt.twinx().plot(hrs[int(2.5 * fps) :], "C2")

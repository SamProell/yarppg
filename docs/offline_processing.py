"""Processing videos offline."""
# %%
import matplotlib.pyplot as plt
import numpy as np

import yarppg

# %%
filename = "../tests/testvideo_30fps.mp4"

fps = yarppg.helpers.get_video_fps(filename)
filter_cfg = yarppg.digital_filter.FilterConfig(fps, 0.5, 1.5, btype="bandpass")
livefilter = yarppg.digital_filter.make_digital_filter(filter_cfg)
processor = yarppg.FilteredProcessor(yarppg.Processor(), livefilter=livefilter)
rppg = yarppg.Rppg(
    processor=processor, hr_calc=yarppg.PeakBasedHrCalculator(fps, window_seconds=4)
)
# %%
results = rppg.process_video(filename)
# %%
values = np.array(results)
hrs = yarppg.bpm_from_frames_per_beat(np.array([res.hr for res in results]), fps)
plt.plot(results[int(2 * fps) :])
# %%

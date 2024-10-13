import pathlib

import yarppg


def test_process_video(testfiles_root: pathlib.Path):
    filename = testfiles_root / "testvideo_30fps.mp4"
    fps = yarppg.get_video_fps(filename)
    filter_cfg = yarppg.digital_filter.FilterConfig(fps, 0.5, 1.5, btype="bandpass")
    livefilter = yarppg.digital_filter.make_digital_filter(filter_cfg)
    processor = yarppg.FilteredProcessor(yarppg.Processor(), livefilter=livefilter)
    hrcalc = yarppg.PeakBasedHrCalculator(fs=fps, window_seconds=5)
    rppg = yarppg.Rppg(processor=processor, hr_calc=hrcalc)

    results = rppg.process_video(filename)

    assert len(results) == 294
    assert abs(yarppg.bpm_from_frames_per_beat(results[-1].hr, fps) - 60) < 2.0

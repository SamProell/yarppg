import pathlib

import yarppg


def test_get_video_fps(testfiles_root: pathlib.Path):
    fps60 = yarppg.helpers.get_video_fps(testfiles_root / "testvideo_60fps.mp4")
    fps30 = yarppg.helpers.get_video_fps(testfiles_root / "testvideo_30fps.mp4")

    assert abs(fps60 - 60) < 0.1
    assert abs(fps30 - 30) < 0.1


def test_frames_from_video(testfiles_root: pathlib.Path):
    filename = testfiles_root / "testvideo_30fps.mp4"

    frame = next(yarppg.helpers.frames_from_video(filename))
    count = sum(1 for _ in yarppg.helpers.frames_from_video(filename))

    assert count == 294
    assert frame.shape == (1080, 1920, 3)

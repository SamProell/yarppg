"""Provides a simplistic user interface with values printed to console only."""
import dataclasses

import cv2

import yarppg


@dataclasses.dataclass
class SimplestOpenCvWindowSettings(yarppg.settings.UiSettings):
    """Configuration for the simplest OpenCV user interface."""

    roi_alpha: float = 0.0
    video: int | str = 0


def launch_loop(rppg: yarppg.Rppg, config: SimplestOpenCvWindowSettings) -> int:
    """Launch a simple Qt6-based GUI visualizing rPPG results in real-time."""
    cam = cv2.VideoCapture(config.video)
    if not cam.isOpened():
        print(f"Could not open {config.video=!r}")
        return -1

    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            break
        result = rppg.process_frame(frame)
        img = yarppg.roi.overlay_mask(
            frame, result.roi.mask != 0, alpha=config.roi_alpha
        )
        print(result.value, result.hr)
        cv2.imshow("frame", cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord("q"):
            break
    return 0

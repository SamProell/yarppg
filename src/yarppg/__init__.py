"""Yet another rPPG implementation."""

__version__ = "1.0"

__all__ = [
    "bpm_from_frames_per_beat",
    "ChromProcessor",
    "Color",
    "DigitalFilter",
    "FaceMeshDetector",
    "FilteredProcessor",
    "frames_from_video",
    "get_config",
    "get_video_fps",
    "HrCalculator",
    "PeakBasedHrCalculator",
    "pixelate_mask",
    "pixelate",
    "Processor",
    "RegionOfInterest",
    "RoiDetector",
    "RppgResult",
    "SelfieDetector",
    "Settings",
    "UiSettings",
]

from .digital_filter import DigitalFilter
from .helpers import bpm_from_frames_per_beat, frames_from_video, get_video_fps
from .hr_calculator import HrCalculator, PeakBasedHrCalculator
from .processors import ChromProcessor, FilteredProcessor, Processor
from .roi import (
    FaceMeshDetector,
    RegionOfInterest,
    RoiDetector,
    SelfieDetector,
    pixelate,
    pixelate_mask,
)
from .rppg import Rppg
from .rppg_result import Color, RppgResult
from .settings import Settings, UiSettings, get_config

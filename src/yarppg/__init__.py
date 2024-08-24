"""Yet another rPPG implementation."""

__version__ = "1.0"

__all__ = [
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
    "pixelate",
    "pixelate_mask",
    "Processor",
    "RegionOfInterest",
    "RoiDetector",
    "RppgResult",
    "SelfieDetector",
    "Settings",
    "ui",
]

from . import ui
from .digital_filter import DigitalFilter
from .helpers import frames_from_video, get_video_fps
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
from .settings import Settings, get_config

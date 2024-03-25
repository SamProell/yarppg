"""A Python implementation of remote photoplethysmography (rPPG)."""

__version__ = "0.2"

import pathlib

from . import ui
from .rppg.camera import Camera
from .rppg.filters import DigitalFilter, FilterConfig, make_digital_filter
from .rppg.hr import HRCalculator, HRCalculatorConfig, make_hrcalculator
from .rppg.processors import FilteredProcessor, ProcessorConfig, get_processor
from .rppg.roi import RegionOfInterest, ROIDetector, ROIDetectorConfig, get_roi_detector
from .rppg.rppg import RPPG, RppgResults

CONFIG_PATH = str(pathlib.Path(__file__).parent / "conf")


__all__ = [
    "Camera",
    "CONFIG_PATH",
    "DigitalFilter",
    "FilterConfig",
    "make_digital_filter",
    "HRCalculator",
    "HRCalculatorConfig",
    "make_hrcalculator",
    "FilteredProcessor",
    "ProcessorConfig",
    "get_processor",
    "RegionOfInterest",
    "ROIDetector",
    "ROIDetectorConfig",
    "get_roi_detector",
    "RPPG",
    "RppgResults",
]

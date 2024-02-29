"""Provides different processors for obtaining the rPPG signal."""
from .chrom import ChromProcessor
from .color_mean import ColorMeanProcessor
from .li_cvpr import LiCvprProcessor
from .pos import PosProcessor
from .processor import FilteredProcessor, Processor

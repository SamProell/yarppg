"""Provides different processors for obtaining the rPPG signal."""
from .chrom import ChromProcessor
from .color_mean import ColorMeanProcessor
from .li_cvpr import LiCvprProcessor
from .pos import PosProcessor
from .processor import FilteredProcessor, Processor, ProcessorConfig


def get_processor(cfg: ProcessorConfig) -> Processor:
    """Initialize rPPG processor."""
    if cfg.name.lower() == "licvpr":
        return LiCvprProcessor(**cfg.kwargs)
    elif cfg.name.lower() == "pos":
        return PosProcessor(**cfg.kwargs)
    elif cfg.name.lower() == "chrom":
        return ChromProcessor(**cfg.kwargs)
    elif cfg.name.lower() == "mean":
        return ColorMeanProcessor(**cfg.kwargs)

    raise NotImplementedError("Configuration not understood: %s", cfg)

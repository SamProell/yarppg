"""Provides different processors for obtaining the rPPG signal."""
from .chrom import ChromProcessor
from .color_mean import ColorMeanProcessor
from .li_cvpr import LiCvprProcessor
from .pos import PosProcessor
from .processor import FilteredProcessor, Processor, ProcessorConfig


def get_processor(cfg: ProcessorConfig) -> Processor:
    """Initialize rPPG processor."""
    if cfg.name.lower() == "licvpr":
        return LiCvprProcessor(winsize=cfg.kwargs.get("winsize", 1))
    elif cfg.name.lower() == "pos":
        return PosProcessor(winsize=cfg.kwargs.get("winsize", 45))
    elif cfg.name.lower() == "chrom":
        return ChromProcessor(
            winsize=cfg.kwargs.get("winsize", 45),
            method=cfg.kwargs.get("method", "xovery"),
        )
    elif cfg.name.lower() == "mean":
        return ColorMeanProcessor(
            winsize=cfg.kwargs.get("winsize", 1), channel=cfg.kwargs.get("channel", "r")
        )

    raise NotImplementedError("Configuration not understood: %s", cfg)

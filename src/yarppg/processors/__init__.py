"""Implementations of various rPPG signal extractors found in literature."""

from typing import Callable

from .chrom import ChromProcessor
from .processor import FilteredProcessor, Processor

algorithms: dict[str, Callable[..., Processor]] = {
    "green": Processor,
    "chrom": ChromProcessor,
}

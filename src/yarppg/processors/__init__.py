"""Implementations of various rPPG signal extractors found in literature.

All processors base the [`Processor`][yarppg.Processor] which most importantly
features a [`process`][yarppg.Processor.process] method.
The `process` function takes an [`RegionOfInterest` container][yarppg.RegionOfInterest]
and extracts the rPPG signal value.

Note that this is a stateful function, for most processors. Many algorithms use an
internal buffer of previous values to provide a more robust calculation.
To clear the internal buffer, we can call [`reset`][yarppg.Processor.reset]

Processors can be wrapped in a [`FilteredProcessor`][yarppg.FilteredProcessor]
allowing for ad-hoc signal smoothing with each signal update.

Besides the base processor, the following additional algorithms from literature
are implemented:

## [ChromProcessor][yarppg.processors.ChromProcessor] (experimental)
Implements the chrominance-based algorithm by
[de Haan, & Jeanne (2013)](https://pubmed.ncbi.nlm.nih.gov/23744659/).

## More to come (your contributions are welcome)
"""

from typing import Callable

from .chrom import ChromProcessor
from .processor import FilteredProcessor, Processor

algorithms: dict[str, Callable[..., Processor]] = {
    "green": Processor,
    "chrom": ChromProcessor,
}

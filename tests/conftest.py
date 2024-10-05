import pathlib

import numpy as np
import pytest

import yarppg


@pytest.fixture
def testfiles_root() -> pathlib.Path:
    """Return the directory containing test files."""
    return pathlib.Path(__file__).parent


@pytest.fixture
def sim_roi() -> yarppg.RegionOfInterest:
    frame = np.arange(16)[:, np.newaxis] * np.arange(16)[np.newaxis, :]
    frame = np.stack([frame, frame, frame], axis=-1)
    frame[..., 1] = 2
    frame[..., 2] = 3

    bg_mask = np.zeros_like(frame[..., 0], dtype="uint8")
    bg_mask[:2] = 1
    frame[bg_mask > 0] = [4, 5, 6]

    mask = np.zeros_like(bg_mask, dtype="uint8")
    mask[4:-4, 3:-3] = 1

    return yarppg.RegionOfInterest(mask, frame.astype("uint8"), bg_mask=bg_mask)

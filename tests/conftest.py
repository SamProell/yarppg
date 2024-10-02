import pathlib

import pytest


@pytest.fixture
def testfiles_root():
    """Return the directory containing test files."""
    return pathlib.Path(__file__).parent

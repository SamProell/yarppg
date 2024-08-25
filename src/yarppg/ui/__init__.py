"""Provides user interfaces for yarppg."""

from typing import Literal, Protocol

from yarppg.rppg import Rppg
from yarppg.settings import Settings


class UiLauncher(Protocol):
    """Function launching a user interface with the given configuration."""

    def __call__(self, rppg: Rppg, settings: Settings) -> int: ...  # noqa: D102


def launch_simple_qt6_ui(rppg: Rppg, settings: Settings) -> int:
    """Launch a simple Qt6-based GUI displaying rPPG results in real time."""
    from yarppg.ui.qt6.main_window import launch_window

    return launch_window(rppg, settings)


launchers = {"qt6": launch_simple_qt6_ui}

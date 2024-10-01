"""Provides user interfaces for yarppg."""

from yarppg.rppg import Rppg

from ..settings import UiSettings


def launch_ui(rppg: Rppg, ui_settings: UiSettings) -> int:
    """Launch a user interface for the given configuration."""
    if type(ui_settings).__name__ == "SimpleQt6WindowSettings":
        from yarppg.ui.qt6.simple_window import SimpleQt6WindowSettings, launch_window

        assert isinstance(ui_settings, SimpleQt6WindowSettings)
        return launch_window(rppg, ui_settings)

    if type(ui_settings).__name__ == "SimplestOpenCvWindowSettings":
        from yarppg.ui.simplest import SimplestOpenCvWindowSettings, launch_loop

        assert isinstance(ui_settings, SimplestOpenCvWindowSettings)
        return launch_loop(rppg, ui_settings)
    raise NotImplementedError(f"Cannot understand the given {ui_settings!r}")

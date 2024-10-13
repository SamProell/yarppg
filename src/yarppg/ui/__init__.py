"""Provides user interfaces for yarPPG.

yarPPG comes with several user interfaces based on additional optional
dependencies. Make sure to install the corresponding extras to use
special UIs, instead of the OpenCV based `simplest` UI.

## Available UIs
### Simplest
This is a simple infinite loop, grabbing new frames from the camera and
visualizing the results in an OpenCV window.

No additional depencies are required for the default interface.


### Simple Qt6 window
A small GUI window highlighting the detected ROI and a trace of the extracted
rPPG signal. Make sure to install extras with:
```bash
pip install ".[qt6]"
```

### More to come
Feel free to contribute other user interfaces using any framework.
"""

import yarppg


def launch_ui(rppg: yarppg.Rppg, ui_settings: yarppg.UiSettings) -> int:
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

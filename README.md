# Welcome to the yarPPG documentation
*yarPPG* is **y**et **a**nother implementation of **r**emote
**P**hoto**P**lethysmo**G**raphy.
Remote photo&shy;plethysmography (rPPG) refers to the camera-based measurement
of a blood volume pulse signal. It works by detecting small changes in skin
color, originating from the pulsation of blood[^1].

> [!CAUTION]
> This is just a hobby project. Intended for demo purposes only, the
  provided program/code is not suitable to be used in a clinical setup
  or for any decision making in general.

> [!IMPORTANT]
> **October 2024 Update** - yarPPG has seen a major overhaul. The rPPG
> processing logic has been completely decoupled from the user interface
> which now also allows offline processing. Not all features have yet
> been ported to the new code base.
>
> Please submit an issue if you miss something from the old version!

## Installation and usage
In order to run the yarPPG application, clone this repository and navigate
to the downloaded folder. You can then install the folder into your Python
environment. This will install the `run-yarppg` command.

```bash
git clone https://github.com/SamProell/yarppg.git
cd yarppg
pip install "."
run-yarppg
```

## Core functionality
Different from earlier versions of yarPPG, the core functionality for remote PPG
signal extraction has been completely decoupled from the user interface.
The `Rppg` class combines all required steps (roi identification, signal extraction,
heart rate estimation) into one (stateful) function.

```python
import yarppg

rppg = yarppg.Rppg()

while running:
    # frame = ...  # get an image array of shape h x w x 3
    result = rppg.process_frame(frame)
    print(f"Current rPPG signal value: {result.value} (HR: {result.hr})")
```

See [this guide] if you need more fine-grained control over the individual
calculation steps.

The `Rppg` class also comes with a method to process an entire video file
at once. See more details [here].

## User interfaces
The default user interface launched by the `run-yarppg` command is a simplistic
window based on OpenCV.
More elaborate user interfaces are available, but require additional dependencies.

### Simple Qt6 window
```bash
pip install ".[qt6]"
run-yarppg ui=qt6_simple
```

### More to come, you are welcome to contribute

[^1]: W Verkruysse, L O Svaasand and J S Nelson. Remote plethysmographic
    imaging using ambient light. *Optics Express*. 2008;16(26):21434–21445.
    doi:[10.1364/oe.16.021434](https://doi.org/10.1364/oe.16.021434)

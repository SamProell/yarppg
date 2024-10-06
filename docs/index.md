# Welcome to the yarPPG documentation
*yarPPG* is **y**et **a**nother implementation of **r**emote
**P**hoto**P**lethysmo**G**raphy.
Remote photo&shy;plethysmography (rPPG) refers to the camera-based measurement
of a blood volume pulse signal. It works by detecting small changes in skin
color, originating from the pulsation of blood[^1].

!!! danger

    This is just a hobby project. Intended for demo purposes only, the
    provided program/code is not suitable to be used in a clinical setup
    or for any decision making in general.


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

[^1]: W Verkruysse, L O Svaasand and J S Nelson. Remote plethysmographic
    imaging using ambient light. *Optics Express*. 2008;16(26):21434–21445.
    doi:[10.1364/oe.16.021434](https://doi.org/10.1364/oe.16.021434)

# Yet another rPPG

> [!WARNING]
> This is a work in progress, do not blindly trust the results. As a demo project, the
> provided program/code is not suitable to be used in a clinical setup or for any
> decision making in general.

> [!NOTE]
> Update 2024: The latest rework of `yarPPG` has only been tested with Python 3.10,
> older versions may not work. `yarPPG` now also uses hydra for the command line
> interface. See the [description](#command-line-options) further below.

## About
`yarPPG` is yet another implementation of remote photoplethysmography in
Python.  Remote photoplethysmography (rPPG) refers to the camera-based
measurement of a (blood volume) pulse signal.  It works by detecting small
changes in skin color, originating from the pulsation of blood [^1].

Some other open-source implementations of rPPG:

* **heartbeat**: Live heart rate measurements. Written in C++ with a number of
  customization options (https://github.com/prouast/heartbeat)
* **Bob's implementation of different rPPG algorithms**: Python implementation
  of different publications for offline evaluation.
  (https://www.idiap.ch/software/bob/docs/bob/bob.rppg.base/master/)

The current default implementation uses the bandpass-filtered green channel
averaged across a custom ROI suggested by Li et al. [^2]. Other features of
their algorithm are a work in progress.

![Screenshot of the yarPPG application](docs/images/yarppg-screenshot.png)

## Installation and usage
In order to run this rPPG implementation, you first have to clone the
repository and make sure all requirements are installed. Afterwards you
can run the yarppg subfolder as a python package using the `-m` flag.

```bash
git clone https://github.com/SamProell/yarppg.git
cd yarppg
pip install -r requirements.txt
python -m yarppg  # from yarppg top-level directory
```

Alternatively, install the package using pip (from the top-level directory).
After installing, you can simply call the provided entry point `run-yarppg`.
```bash
pip install -e .
run-yarppg
```

### Command line options
There are a number of options available, when running yarPPG.

```yaml
video: 0  # camera index or video filename.
blur: -1  # pixelate the detected face (size of pixelated pixels).
savepath: null  # path where to store the history of detected rPPG signals.
delay_ms: null  # delay the next frame by x milliseconds. Necessary when playing videos
hrcalc:  # options for HR calculation
  update_interval: 30  # update every N frames
  winsize: 300  # number of samples to consider for calculation (~10 seconds)
  filt:
    fs: 30.0  # corresponds to FPS of camera
    f1: 1.5   # first cutoff (for lowpass, highpass or bandpass)
    f2: null  # second cutoff (required for bandpass)
    btype: low  # filter type
    ftype: butter  # filter implementation
    order: 2
roidetect:  # Region of interest detector
  name: facemesh  # other options are 'haar', 'caffednn', 'full'
  kwargs: {}  # additional keyword arguments (sensible defaults are configured already)
processor:  # Method to calculate heart beat signal from the ROI
  name: Mean # 'Chrom', 'Pos', 'LiCvpr' (work in progress)
  kwargs:
    channel: g
filt:  # filter detected heart beat signal, set to null to disable
  fs: 30.0
  f1: 0.4
  f2: 2.0
  btype: band
  ftype: butter
  order: 2
```

To modify the settings at runtime, you can use
[Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)
Below are a few examples:

```bash
# run yarppg with the default Caffe DNN face detector
run-yarppg roidetect.name=caffednn

# read video file instead of webcam, adjust the delay between frames to get proper FPS
run-yarppg video=path/to/video.mp4 delay_ms=30

# pixelate the face; update HR every 5 frames; use POS processor with custom args;
run-yarppg blur=20 hrcalc.update_interval=5 processor.name=pos processor.kwargs="{winsize:30}"
```

### Camera setup
For rPPG to work as intended it might be required to tweak the camera
settings. For example auto white-balancing and auto-exposure could be
disabled, to avoid small adjustments in RGB values.
One possibility to change the camera settings is via the ``ffmpeg``
command line tool.
See [this discussion](https://superuser.com/questions/1287366/open-webcam-settings-dialog-in-windows/1511657)
for more details.

[^1]: W Verkruysse, L O Svaasand and J S Nelson. Remote plethysmographic
      imaging using ambient light. *Optics Express*. 2008;16(26):21434–21445.
      doi:[10.1364/oe.16.021434](https://doi.org/10.1364/oe.16.021434)

[^2]: Li, X., Chen, J., Zhao, G., &#38; Pietikainen, M. (2014). Remote
      Heart Rate Measurement From Face Videos Under Realistic Situations.
      Proceedings of the IEEE Conference on Computer Vision and Pattern
      Recognition (CVPR), 4264-4271.
      doi:[10.1109/CVPR.2014.543](https://doi.org/10.1109/CVPR.2014.543)


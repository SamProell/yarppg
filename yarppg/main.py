from ast import parse
import sys
import argparse


from PyQt5.QtWidgets import QApplication
from yarppg.rppg.camera import Camera
from yarppg.ui import MainWindow
from yarppg.rppg import RPPG
from yarppg.rppg.processors import ColorMeanProcessor, FilteredProcessor
from yarppg.rppg.hr import HRCalculator
from yarppg.rppg.filters import get_butterworth_filter
from yarppg.ui.cli import (get_detector, get_mainparser, get_processor,
                           parse_frequencies, get_delay)


def main():
    parser = get_mainparser()
    args = parser.parse_args(sys.argv[1:])
    app = QApplication(sys.argv)

    roi_detector = get_detector(args)

    digital_lowpass = get_butterworth_filter(30, 1.5)
    hr_calc = HRCalculator(parent=app, update_interval=30, winsize=300,
                           filt_fun=lambda vs: [digital_lowpass(v) for v in vs])

    processor = get_processor(args)

    cutoff = parse_frequencies(args.bandpass)
    if cutoff is not None:
        digital_bandpass = get_butterworth_filter(30, cutoff, "bandpass")
        processor = FilteredProcessor(processor, digital_bandpass)

    cam = Camera(video=args.video, limit_fps=get_delay(args))
    rppg = RPPG(roi_detector=roi_detector,
                camera=cam,
                hr_calculator=hr_calc,
                parent=None,
                )
    rppg.add_processor(processor)
    for c in "rgb":
        rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1))

    if args.savepath:
        rppg.output_filename = args.savepath

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(1000, 400),
                     legend=True,
                     graphwin=300,
                     blur_roi=args.blur,
                     )
    for i in range(3):
        win.set_pen(index=i+1, color="rgb"[i], width=1)

    return win.execute()


if __name__ == "__main__":
    sys.exit(main())

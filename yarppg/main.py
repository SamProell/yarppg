"""Main functionality of the rPPG application."""
import sys
from dataclasses import dataclass

from PyQt5.QtWidgets import QApplication

from yarppg.rppg import RPPG
from yarppg.rppg.camera import Camera
from yarppg.rppg.filters import FilterConfig, get_butterworth_filter
from yarppg.rppg.hr import HRCalculatorConfig, make_hrcalculator
from yarppg.rppg.processors import ColorMeanProcessor, FilteredProcessor
from yarppg.ui import MainWindow
from yarppg.ui.cli import (
    get_delay,
    get_detector,
    get_mainparser,
    get_processor,
    parse_frequencies,
)


@dataclass
class Settings:
    video: int = 0
    hrcalc: HRCalculatorConfig = HRCalculatorConfig(
        update_interval=30,
        winsize=300,
        filt=FilterConfig(
            fs=30,
            f1=1.5,
        ),
    )


def main(cfg: Settings = Settings()):
    """Run the rPPG application."""
    parser = get_mainparser()
    args = parser.parse_args(sys.argv[1:])
    app = QApplication(sys.argv)

    roi_detector = get_detector(args)

    hr_calc = make_hrcalculator(cfg.hrcalc, parent=app)

    processor = get_processor(args)

    cutoff = parse_frequencies(args.bandpass)
    if cutoff is not None:
        digital_bandpass = get_butterworth_filter(30, cutoff, "bandpass")
        processor = FilteredProcessor(processor, digital_bandpass)

    cam = Camera(video=cfg.video, limit_fps=get_delay(args))
    rppg = RPPG(
        roi_detector=roi_detector,
        camera=cam,
        hr_calculator=hr_calc,
        parent=None,
    )
    rppg.add_processor(processor)
    for c in "rgb":
        rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1))

    if args.savepath:
        rppg.output_filename = args.savepath

    win = MainWindow(
        app=app,
        rppg=rppg,
        winsize=(1000, 400),
        legend=True,
        graphwin=300,
        blur_roi=args.blur,
    )
    for i in range(3):
        win.set_pen(index=i + 1, color="rgb"[i], width=1)

    return win.execute()


if __name__ == "__main__":
    sys.exit(main())

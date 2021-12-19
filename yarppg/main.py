import sys
import argparse


from PyQt5.QtWidgets import QApplication
from yarppg.ui import MainWindow
from yarppg.rppg import RPPG
from yarppg.rppg.roi_detect import CaffeDNNFaceDetector, NoDetector
from yarppg.rppg.processors import (ColorMeanProcessor, ChromProcessor,
                                    FilteredProcessor)
from yarppg.rppg.hr import HRCalculator, from_fft
from yarppg.rppg.filters import DigitalFilter, get_butterworth_filter


_mainparser = argparse.ArgumentParser(description="Use your Webcam to measure"
                                                  "your heart rate")
_mainparser.add_argument("--blobsize", default=150, type=int,
                         help="quadratic blob size of DNN Face Detector")
_mainparser.add_argument("--blur", default=25, type=int,
                         help="pixelation size of detected ROI")
_mainparser.add_argument("--video", default=0, type=int,
                         help="video input device number")

def main():
    args = _mainparser.parse_args(sys.argv[1:])
    app = QApplication(sys.argv)

    # roi_detector = NoDetector()
    roi_detector = CaffeDNNFaceDetector(blob_size=(args.blobsize, args.blobsize),
                                        smooth_factor=0.9)

    digital_lowpass = get_butterworth_filter(30, 1.5)
    digital_bandpass = get_butterworth_filter(30, cutoff=(0.5, 4),
                                              btype="bandpass")
    hr_calc = HRCalculator(parent=app, update_interval=30, winsize=300,
                           filt_fun=lambda vs: [digital_lowpass(v) for v in vs])

    rppg = RPPG(roi_detector=roi_detector,
                video=args.video,
                hr_calculator=hr_calc,
                parent=app,
                )
    processor = ChromProcessor(winsize=5, method="xovery")
    rppg.add_processor(FilteredProcessor(processor, digital_bandpass))
    rppg.add_processor(ColorMeanProcessor(channel="r", winsize=1))
    rppg.add_processor(ColorMeanProcessor(channel="g", winsize=1))
    rppg.add_processor(ColorMeanProcessor(channel="b", winsize=1))

    # rppg.output_filename = "outputs.csv"

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

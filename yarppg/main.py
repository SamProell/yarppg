from ast import parse
import sys
import argparse


from PyQt5.QtWidgets import QApplication
from yarppg.ui import MainWindow
from yarppg.rppg import RPPG
from yarppg.rppg.roi.roi_detect import CaffeDNNFaceDetector, FaceMeshDetector, HaarCascadeDetector, NoDetector
from yarppg.rppg.processors import (ColorMeanProcessor, ChromProcessor,
                                    FilteredProcessor, PosProcessor, LiCvprProcessor)
from yarppg.rppg.hr import HRCalculator, from_fft
from yarppg.rppg.filters import DigitalFilter, get_butterworth_filter


_mainparser = argparse.ArgumentParser(description="Use your Webcam to measure"
                                                  "your heart rate")
_mainparser.add_argument("--detector", default="facemesh", type=str,
                         choices=["facemesh", "caffe-dnn", "haar", "full"],
                         help="ROI (face) detector")
_mainparser.add_argument("--processor", default="LiCvpr",
                         choices=["LiCvpr", "Pos", "Chrom"],
                         help=("Processor translating ROI to pulse signal. "
                               "LiCvpr currently only returns mean green value"))
_mainparser.add_argument("--winsize", default=32, type=int,
                         help="Window sized used in some processors")
_mainparser.add_argument("--bandpass", type=str, default="0.5,2",
                         help="bandpass frequencies for processor output")
_mainparser.add_argument("--blobsize", default=150, type=int,
                         help="quadratic blob size of DNN Face Detector")
_mainparser.add_argument("--draw-facemark", action="store_true",
                         help="draw landmarks when using facemesh detector")
_mainparser.add_argument("--blur", default=-1, type=int,
                         help="pixelation size of detected ROI")
_mainparser.add_argument("--video", default=0, type=int,
                         help="video input device number")
_mainparser.add_argument("--savepath", default="", type=str,
                         help="store generated signals as data frame")



def get_detector(args):
    name = args.detector.lower()

    if name == "full":
        return NoDetector()
    elif name == "facemesh":
        return FaceMeshDetector(draw_landmarks=args.draw_facemark)
    elif name == "caffe-dnn":
        return CaffeDNNFaceDetector(blob_size=args.blobsize)
    elif name == "haar":
        return HaarCascadeDetector()

    raise NotImplementedError(f"detector {args.detector!r} not recognized.")


def get_processor(args):
    name = args.processor.lower()
    if name == "licvpr":
        return LiCvprProcessor()
    elif name == "pos":
        return PosProcessor(winsize=args.winsize)
    elif name == "chrom":
        return ChromProcessor(winsize=args.winsize, method="xovery")


def parse_frequencies(s):
    """Rudimentary parser of frequency string expected as 'f1,f2' (e.g. '0.4,2')
    """
    if s == "none":
        return None
    return list(map(float, s.split(",")))


def main():
    args = _mainparser.parse_args(sys.argv[1:])
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


    rppg = RPPG(roi_detector=roi_detector,
                video=args.video,
                hr_calculator=hr_calc,
                parent=app,
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

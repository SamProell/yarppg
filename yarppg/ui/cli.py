"""Defines command line options and corresponding initialization methods."""
import argparse

from yarppg.rppg.processors import ChromProcessor, LiCvprProcessor, PosProcessor


def get_mainparser():
    """Define the argument parser for the main script."""
    parser = argparse.ArgumentParser(
        description="Use your Webcam to measure" "your heart rate"
    )
    parser.add_argument(
        "--detector",
        default="facemesh",
        type=str,
        choices=["facemesh", "caffe-dnn", "haar", "full"],
        help="ROI (face) detector",
    )
    parser.add_argument(
        "--processor",
        default="LiCvpr",
        choices=["LiCvpr", "Pos", "Chrom"],
        help=(
            "Processor translating ROI to pulse signal. "
            "LiCvpr currently only returns mean green value"
        ),
    )
    parser.add_argument(
        "--winsize", default=32, type=int, help="Window sized used in some processors"
    )
    parser.add_argument(
        "--bandpass",
        type=str,
        default="0.5,2",
        help="bandpass frequencies for processor output",
    )
    parser.add_argument(
        "--blobsize",
        default=150,
        type=int,
        help="quadratic blob size of DNN Face Detector",
    )
    parser.add_argument(
        "--draw-facemark",
        action="store_true",
        help="draw landmarks when using facemesh detector",
    )
    parser.add_argument(
        "--blur", default=-1, type=int, help="pixelation size of detected ROI"
    )
    parser.add_argument("--video", default=0, help="video input device number")
    parser.add_argument(
        "--savepath", default="", type=str, help="store generated signals as data frame"
    )
    parser.add_argument(
        "--limitfps", type=float, default=None, help="limit FPS to specified maximum"
    )
    parser.add_argument(
        "--delay-frames",
        type=float,
        default=None,
        help=(
            "add a delay of specified number of milliseconds" " (overrides --limitfps)"
        ),
    )

    return parser


def get_delay(args):
    """Get delay for video mode."""
    if args.delay_frames is not None:
        return 1000.0 / args.delay_frames

    return args.limitfps


def get_processor(args):
    """Get processor with given arguments."""
    name = args.processor.lower()
    if name == "licvpr":
        return LiCvprProcessor()
    elif name == "pos":
        return PosProcessor(winsize=args.winsize)
    elif name == "chrom":
        return ChromProcessor(winsize=args.winsize, method="xovery")


def parse_frequencies(s):
    """Parser of frequency string expected as 'f1,f2' (e.g. '0.4,2')."""
    if s == "none":
        return None
    return list(map(float, s.split(",")))

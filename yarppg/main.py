import sys


def main():
    from PyQt5.QtWidgets import QApplication

    from yarppg.ui import MainWindow
    from yarppg.rppg import RPPG
    from yarppg.rppg.roi_detect import CaffeDNNFaceDetector
    from yarppg.rppg.processors import (ColorMeanProcessor, ChromProcessor,
                                        FilteredProcessor)
    from yarppg.rppg.hr import HRCalculator
    from yarppg.rppg.filters import DigitalFilter, get_butterworth_filter

    app = QApplication(sys.argv)
    roi_detector = CaffeDNNFaceDetector(blob_size=(150, 150))

    digital_lowpass = get_butterworth_filter(30, 1.5)
    hr_calc = HRCalculator(parent=app, update_interval=30, winsize=200)

    rppg = RPPG(roi_detector=roi_detector,
                roi_smooth=0.9,
                video=0,
                hr_calculator=hr_calc,
                parent=app,
                )
    processor = ChromProcessor(winsize=5, method="xovery")
    rppg.add_processor(FilteredProcessor(processor, digital_lowpass))
    rppg.add_processor(ColorMeanProcessor(channel="r", winsize=1))
    rppg.add_processor(ColorMeanProcessor(channel="g", winsize=1))
    rppg.add_processor(ColorMeanProcessor(channel="b", winsize=1))

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(1000, 400),
                     legend=True,
                     graphwin=300,
                     blur_roi=25,
                     )
    for i in range(3):
        win.set_pen(index=i+1, color="rgb"[i], width=1)

    return win.execute()


if __name__ == "__main__":
    exit(main())

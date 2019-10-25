import sys

from PyQt5.QtWidgets import QApplication

from app import MainWindow, RPPG, HaarCascadeDetector, CaffeDNNFaceDetector
from app.rppg.processors import ColorMeanProcessor, ChromProcessor

if __name__ == "__main__":
    app = QApplication(sys.argv)
    roi_detector = CaffeDNNFaceDetector(blob_size=(150, 150))

    rppg = RPPG(roi_detector=roi_detector,
                smooth_roi=0.9,
                video=0,
                parent=app,
                )
    winsize = 10
    rppg.add_processor(ColorMeanProcessor(channel="r", winsize=1))
    rppg.add_processor(ChromProcessor(winsize=winsize))
    rppg.add_processor(ChromProcessor(winsize=winsize, method="fixed"))

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(1000, 400),
                     legend=True,
                     )
    for i in range(2):
        win.set_pen(index=i+1, color="grb"[i], width=1)
    sys.exit(win.execute())

import sys

from PyQt5.QtWidgets import QApplication

from app import MainWindow, RPPG, HaarCascadeDetector, CaffeDNNFaceDetector
from app.rppg.processors import ColorMeanProcessor, ChromProcessor

if __name__ == "__main__":
    app = QApplication(sys.argv)
    roi_detector = CaffeDNNFaceDetector(blob_size=(100, 100))

    rppg = RPPG(roi_detector=roi_detector,
                smooth_roi=0.9,
                video=0,
                parent=app,
                )
    rppg.add_processor(ColorMeanProcessor(channel="g"))
    rppg.add_processor(ColorMeanProcessor(channel="g"))
    rppg.add_processor(ChromProcessor(winsize=45))

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(800, 300)
                     )

    sys.exit(win.execute())

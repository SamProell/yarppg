import sys

from PyQt5.QtWidgets import QApplication

from app import MainWindow, RPPG, HaarCascadeDetector, CaffeDNNFaceDetector

if __name__ == "__main__":
    app = QApplication(sys.argv)
    roi_detector = CaffeDNNFaceDetector(blob_size=(150, 150))

    rppg = RPPG(roi_detector=roi_detector,
                smooth_roi=0.9,
                video=0,
                parent=app,
                )
    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(300, 200)
                     )

    sys.exit(win.execute())

import sys


def main():
    from PyQt5.QtWidgets import QApplication

    from yarppg.ui import MainWindow
    from yarppg.rppg import RPPG
    from yarppg.rppg.roi_detect import HaarCascadeDetector, CaffeDNNFaceDetector
    from yarppg.rppg.processors import ColorMeanProcessor, ChromProcessor

    app = QApplication(sys.argv)
    roi_detector = CaffeDNNFaceDetector(blob_size=(150, 150))

    rppg = RPPG(roi_detector=roi_detector,
                roi_smooth=0.9,
                video=0,
                parent=app,
                )
    rppg.add_processor(ChromProcessor(winsize=5, method="xovery"))
    rppg.add_processor(ColorMeanProcessor(channel="r", winsize=1))
    rppg.add_processor(ColorMeanProcessor(channel="g", winsize=1))
    rppg.add_processor(ColorMeanProcessor(channel="b", winsize=1))

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(1000, 400),
                     legend=True,
                     graphwin=300,
                     )
    for i in range(3):
        win.set_pen(index=i+1, color="rgb"[i], width=1)

    return win.execute()


if __name__ == "__main__":
    exit(main())

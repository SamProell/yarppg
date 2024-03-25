"""Main functionality of the rPPG application."""

import sys
from dataclasses import dataclass
from typing import Optional, Union

import hydra
import hydra.core.config_store
from PyQt5.QtWidgets import QApplication

import yarppg


# The below implementation is not 100% technically correct.  Default factories are
# neglected to highlight the structure of the configuration and set sensible defaults.
@dataclass
class Settings:
    video: Union[int, str] = 0
    blur: int = -1
    savepath: Optional[str] = None
    delay_ms: Optional[float] = None
    hrcalc: yarppg.HRCalculatorConfig = yarppg.HRCalculatorConfig(
        update_interval=30,
        winsize=300,
        filt=yarppg.FilterConfig(
            fs=30,
            f1=1.5,
        ),
    )
    roidetect: yarppg.ROIDetectorConfig = yarppg.ROIDetectorConfig(
        name="facemesh",
    )
    processor: yarppg.ProcessorConfig = yarppg.ProcessorConfig(
        "Mean",
        kwargs={
            "winsize": 1,
            "channel": "g",
        },
    )
    filt: Optional[yarppg.FilterConfig] = yarppg.FilterConfig(
        fs=30,
        f1=0.4,
        f2=2,
        btype="band",
    )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store("yarppg_schema", node=Settings)


@hydra.main(yarppg.CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: Settings):
    """Run the rPPG application."""
    app = QApplication(sys.argv)

    roi_detector = yarppg.get_roi_detector(cfg.roidetect)
    hr_calc = yarppg.make_hrcalculator(cfg.hrcalc, parent=app)

    processor = yarppg.get_processor(cfg.processor)
    if cfg.filt is not None:
        digital_filter = yarppg.make_digital_filter(cfg.filt)
        processor = yarppg.FilteredProcessor(processor, digital_filter)

    cam = yarppg.rppg.Camera(video=cfg.video, limit_fps=cfg.delay_ms)  # type: ignore

    rppg = yarppg.RPPG(
        roi_detector=roi_detector, camera=cam, hr_calculator=hr_calc, parent=None
    )
    rppg.add_processor(processor)
    for c in "rgb":
        proc = yarppg.get_processor(
            yarppg.ProcessorConfig("Mean", {"channel": c, "winsize": 1})
        )
        rppg.add_processor(proc)

    if cfg.savepath is not None:
        rppg.output_filename = cfg.savepath

    win = yarppg.ui.MainWindow(
        app, rppg, winsize=(1000, 400), legend=True, graphwin=300, blur_roi=cfg.blur
    )
    for i in range(3):
        win.set_pen(index=i + 1, color="rgb"[i], width=1)

    return win.execute()


if __name__ == "__main__":
    main()

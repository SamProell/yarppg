from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class Settings:
    video: int | str = 0
    blursize: int | None = None
    savepath: str | None = None
    frame_delay: float = float("nan")
    roi_alpha: float = 0.0
    ui: str = "qt6"


def get_config(argv: list[str]) -> Settings:
    default_config = OmegaConf.structured(Settings)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())
    return config  # type: ignore

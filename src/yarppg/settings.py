from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class Settings:
    video: int | str = 0
    blursize: int | None = None
    savepath: str | None = None
    frame_delay: float = float("nan")
    roi_alpha: float = 0.0
    # backend: str = "qt6"


def get_config(argv: list[str]) -> OmegaConf:
    default_config = OmegaConf.structured(Settings)
    settings = OmegaConf.merge(default_config, OmegaConf.from_cli(argv))
    return settings  # type: ignore

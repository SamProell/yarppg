"""Provides configuration containers for the yarppg application."""

from dataclasses import dataclass

from omegaconf import OmegaConf

from .digital_filter import FilterConfig


@dataclass
class Settings:
    """Comprises all configuration options available in the yarppg application."""

    video: int | str = 0
    blursize: int | None = None
    savepath: str | None = None
    frame_delay: float = float("nan")
    roi_alpha: float = 0.0
    ui: str = "qt6"
    detector: str = "facemesh"
    filter: FilterConfig | None = None
    algorithm: str = "green"


def get_config(argv: list[str] | None = None) -> Settings:
    """Build the configuration from default values and the command line."""
    default_config = OmegaConf.structured(Settings)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli(argv))
    return config  # type: ignore

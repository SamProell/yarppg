"""Provides configuration containers for the yarPPG application."""

# import copy
import dataclasses  # import dataclass, field
from typing import Any

import hydra.conf
import hydra.core.config_store
import hydra.utils

from .digital_filter import FilterConfig


@dataclasses.dataclass
class UiSettings:
    """Settings for the user interface."""


@dataclasses.dataclass(kw_only=True)
class HydraSettings:
    """Base class for Hydra-based configurations.

    Mainly manages the hydra-specific settings, deactivating its directory and output
    management, so that running with hydra.main does not behave differently from a
    normal CLI.
    """

    hydra: "hydra.conf.HydraConf" = dataclasses.field(
        default_factory=lambda: hydra.conf.HydraConf(
            output_subdir=None,
            run=hydra.conf.RunDir("."),
            help=hydra.conf.HelpConf(app_name="run-yarppg"),
            overrides=hydra.conf.OverridesConf(
                # hydra=["job_logging=null", "hydra_logging=null"]
            ),
        )
    )


@dataclasses.dataclass
class Settings(HydraSettings):
    """Comprises all configuration options available in the yarppg application."""

    ui: Any
    savepath: str | None = None
    detector: str = "facemesh"
    filter: FilterConfig | None = dataclasses.field(
        default_factory=lambda: FilterConfig(30, 0.5, 2, btype="bandpass")
    )
    algorithm: str = "green"
    defaults: Any = dataclasses.field(
        default_factory=lambda: [
            {"ui": "simplest"},
            "_self_",
        ]
    )


def available_ui_configs():
    """Check availability of UIs and return each corresponding settings container."""
    import yarppg.ui.simplest

    uis: dict[str, Any] = {"simplest": yarppg.ui.simplest.SimplestOpenCvWindowSettings}
    try:
        import yarppg.ui.qt6

        uis["qt6_simple"] = yarppg.ui.qt6.SimpleQt6WindowSettings
    except (ModuleNotFoundError, ImportError):
        pass

    return uis


def register_schemas():
    """Register base schema and settings for available UI implementations."""
    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name="config", node=Settings)
    for name, cfg_class in available_ui_configs().items():
        cs.store(name=name, node=cfg_class, group="ui")


def get_config(argv: list[str] | None) -> Settings:
    """Get the default configuration with optional overrides."""
    register_schemas()
    with hydra.initialize():
        cfg = hydra.compose(config_name="config", overrides=argv)
    return hydra.utils.instantiate(cfg)

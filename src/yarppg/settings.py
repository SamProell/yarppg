"""Provides configuration containers for the yarppg application."""

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


@dataclasses.dataclass
class Settings:
    """Comprises all configuration options available in the yarppg application."""

    ui: Any
    savepath: str | None = None
    detector: str = "facemesh"
    filter: FilterConfig | None = dataclasses.field(
        default_factory=lambda: FilterConfig(30, 0.5, 2, btype="bandpass")
    )
    algorithm: str = "green"
    hydra: "hydra.conf.HydraConf" = dataclasses.field(
        default_factory=lambda: hydra.conf.HydraConf(
            output_subdir=None,
            run=hydra.conf.RunDir("."),
            overrides=hydra.conf.OverridesConf(
                # hydra=["job_logging=null", "hydra_logging=null"]
            ),
        )
    )
    defaults: Any = dataclasses.field(
        default_factory=lambda: [
            {"ui": "simplest"},
            "_self_",
        ]
    )


def available_ui_configs():
    import yarppg.ui.simplest

    uis: dict[str, Any] = {"simplest": yarppg.ui.simplest.SimplestOpenCvWindowSettings}
    try:
        import yarppg.ui.qt6

        uis["qt6_simple"] = yarppg.ui.qt6.SimpleQt6WindowSettings
    except (ModuleNotFoundError, ImportError):
        pass

    return uis


def register_schemas():
    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name="config", node=Settings)
    for name, cfg_class in available_ui_configs().items():
        cs.store(name=name, node=cfg_class, group="ui")


def get_config(argv: list[str]) -> Settings:
    register_schemas()
    with hydra.initialize():
        cfg = hydra.compose(config_name="config", overrides=argv)
    return hydra.utils.instantiate(cfg)


# def get_config(argv: list[str]) -> Settings:
#     """Build the configuration from default values and the command line."""
#     default_config = omegaconf.OmegaConf.structured(Settings)
#     config = omegaconf.OmegaConf.merge(
#         default_config, omegaconf.OmegaConf.from_cli(argv)
#     )
#     return omegaconf.OmegaConf.to_object(config)  # type: ignore


# def flatten_dict(cfg: dict, parent_key="", sep="."):
#     """Recursively flattens a nested dictionary, concatenating keys with a separator."""
#     items = []
#     for k, v in cfg.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)


# def print_help():
#     """Prints a help message and all options available through the Settings class."""
#     print("yarPPG demonstrates remote Photoplethysmography with your own camera.\n")
#     print("Options:")
#     config_dict = OmegaConf.to_container(OmegaConf.structured(Settings))
#     for key, value in flatten_dict(config_dict).items():  # type: ignore
#         print(f"{key:>20}: {value}")

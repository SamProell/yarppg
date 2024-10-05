"""Main entrypoint for yarppg GUI."""

import hydra
import omegaconf

import yarppg
import yarppg.ui


@hydra.main(version_base=None, config_name="config")
def main(cfg: omegaconf.DictConfig):
    """Initialize the an rPPG orchestrator with CLI arguments and launch UI."""
    config: yarppg.Settings = omegaconf.OmegaConf.to_object(cfg)  # type: ignore
    rppg = yarppg.Rppg.from_settings(config)

    yarppg.ui.launch_ui(rppg, config.ui)


def run_yarppg():
    """Register structured configs and run the main function."""
    yarppg.settings.register_schemas()
    main()


if __name__ == "__main__":
    run_yarppg()

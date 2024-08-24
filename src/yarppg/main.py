"""Main entrypoint for yarppg GUI."""

import sys

import yarppg


def main():
    """Initialize the an rPPG orchestrator with CLI arguments and launch UI."""
    config = yarppg.get_config(sys.argv)
    rppg = yarppg.Rppg.from_settings(config)

    yarppg.ui.launchers[config.ui](rppg, config)


if __name__ == "__main__":
    main()

"""Copyright (c) 2022, Liberty Mutual Group."""
import logging
import logging.config
import os

import yaml

from .constants import BASE_PATH

LOGLEVEL_PLACEHOLDER = "__LOGLEVEL__"

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging.

    Load in the logging.yml file, and replace the LOGLEVEL placeholder with the value from the environment variable.
    """
    try:
        with open(BASE_PATH / "pipeline" / "config" / "logging.yml", "r") as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

            # Replace {LOGLEVEL} placeholder with value from environment variable
            for logger in config["loggers"]:
                if LOGLEVEL_PLACEHOLDER in config["loggers"][logger]["level"]:
                    config["loggers"][logger]["level"] = os.environ.get("LOGLEVEL", "DEBUG")

            if "root" in config and "level" in config["root"]:
                if LOGLEVEL_PLACEHOLDER in config["root"]["level"]:
                    config["root"]["level"] = os.environ.get("LOGLEVEL", "DEBUG")

            # Configure logging
            logging.config.dictConfig(config)
    except FileNotFoundError:
        LOGGER.debug("logging.yml not found. Skipping logging configuration.")

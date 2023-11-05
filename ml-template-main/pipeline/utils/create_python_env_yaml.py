"""Copyright (c) 2023, Liberty Mutual Group."""
import logging
import os

import yaml
from mlflow.utils.environment import _PYTHON_ENV_FILE_NAME, _PythonEnv

from ..config.constants import BASE_PATH

logger = logging.getLogger(__name__)


def create_python_env_yaml() -> None:
    """Train and log the model to MLFlow."""
    _PythonEnv.current().to_yaml(os.path.join(BASE_PATH, _PYTHON_ENV_FILE_NAME))

    # Modifiy dependencies to point to requirements_generated.txt
    with open(BASE_PATH / _PYTHON_ENV_FILE_NAME, "r") as python_env_yaml:
        python_env = yaml.safe_load(python_env_yaml)
        python_env["dependencies"] = ["-r requirements_generated.txt"]

    # Write updated file
    with open(BASE_PATH / _PYTHON_ENV_FILE_NAME, "w") as python_env_yaml_updated:
        yaml.dump(python_env, python_env_yaml_updated)


if __name__ == "__main__":
    logger.info("Creating python_env.yaml...")
    create_python_env_yaml()

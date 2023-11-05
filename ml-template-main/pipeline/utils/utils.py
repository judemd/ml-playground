import logging
import os
import pickle
import re
import subprocess
import tempfile
from configparser import ConfigParser
from pathlib import Path
from sys import platform
from typing import Any, List

import mlflow
import requests
from lit_ds_utils.git_utils import repo_is_clean
from lit_ds_utils.mlflow_utils import is_use_managed_mlflow

from pipeline import settings
from pipeline.config.constants import BASE_PATH, ENFORCE_CLEAN_WORKSPACE

logger = logging.getLogger(__name__)

EXCLUDE_PKGS_CONFIG_OPTION = "EXCLUDE_PACKAGES_FROM_REQUIREMENTS_TXT"
DEFAULT_PACKAGES_TO_EXCLUDE = ["setuptools"]


def generate_requirements_txt(packages_to_exclude: List[str] = None) -> None:
    """Generate the requirements_generated.txt file in the base directory, from the pipenv environment.

    Args:
        packages_to_exclude: The list of packages to exclude from the generated file. Optional.

    Raises:
        Exception: If an error occurs running "pipenv requirements"
    """
    env_vars = os.environ
    env_vars["PIPENV_VERBOSITY"] = "-1"  # Suppress warning message about running in a virtual environment

    # If no explicit packages to exclude, check the config
    settings_ini = BASE_PATH / "settings.ini"

    config_object = ConfigParser()
    config_object.read(settings_ini)

    default_section = config_object["default"]

    if packages_to_exclude is None:
        # If argument not specified, check in the config file
        if EXCLUDE_PKGS_CONFIG_OPTION in default_section:
            packages_to_exclude = default_section[EXCLUDE_PKGS_CONFIG_OPTION].split(",")

    if packages_to_exclude is None:
        packages_to_exclude = DEFAULT_PACKAGES_TO_EXCLUDE

    logger.info(f"Packages to exclude: {packages_to_exclude}")

    # create two files to hold the output and errors, respectively
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as fout:
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as ferr:
            subprocess.check_call(
                ["pipenv", "requirements"],
                stdout=fout,
                stderr=ferr,
                cwd=BASE_PATH,
                env=env_vars,
            )
            # reset file to read from it
            fout.seek(0)
            # save output (if any) in variable
            output = fout.read()
            fout.seek(0)

            if not output:
                raise Exception("Empty output from 'pipenv requirements'")

            # reset file to read from it
            ferr.seek(0)
            # save errors (if any) in variable
            errors = ferr.read()

            if errors:
                logger.warning(f"Errors returned from 'pipenv requirements': {errors}")

        # Copy lines over to requirements_generated.txt, with exclusions
        with open(BASE_PATH / "requirements_generated.txt", "w+") as requirements_txt:
            for line in fout.readlines():
                write_line = True
                for p in packages_to_exclude:
                    # Search for package name, followed by optional spaces and a double equals at the start of the line
                    if re.match(f"^{p}\\s*==", line):
                        write_line = False
                        break
                if write_line:
                    requirements_txt.write(line)


def save_local_artifact(artifact_path: Path, artifact: Any) -> None:
    """Saves off an artifact by pickling it.

    Args:
        artifact_path: The artifact path.
        artifact: The artifact to be saved.
    """
    # Store input data
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)


def load_local_artifact(artifact_path: Path) -> Any:
    """Load a previously saved artifact.

    Args:
        artifact_path: The artifact.

    Returns:
        The unpickled artifact.
    """
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def check_for_clean_repository() -> None:
    """Check the git repository is clean.

    If the ENFORCE_CLEAN_WORKSPACE is set to True, and the repository has been modified, log and error and exit
    the program.
    """
    if not repo_is_clean(Path(__file__).parent, perform_git_fetch=True):
        if settings.bool(ENFORCE_CLEAN_WORKSPACE, False):
            logger.critical(
                "Repository contains uncommitted files, and ENFORCE_CLEAN_WORKSPACE flag is True. Exiting.."
            )
            exit(1)
        else:
            logger.warning("Repository contains uncommitted files.")


def check_for_expired_databricks_token() -> None:
    """Check that the current databricks token has not expired.

    If using managed MLFlow (Databricks), make a request to the API with the current token, to ensure that the token is
    still valid. Exit the program if the API call fails.
    """
    if is_use_managed_mlflow():
        logger.info("Checking databricks token")
        # Attempt a call to get the list of registered models
        url = f"{settings.str('DATABRICKS_HOST').strip('/')}/{settings.str('MLFLOW_API_CONTEXT_ROOT')}/registered-models/list"
        databricks_response = requests.get(url, headers={"Authorization": f'Bearer {settings.str("DATABRICKS_TOKEN")}'})
        if databricks_response.status_code != 200:
            logger.critical(
                "Failed to communicate with Databricks. Status code: %s. The databricks token may have expired.",
                databricks_response.status_code,
            )
            exit(1)
        else:
            logger.info("Databricks token is valid")


def check_for_secure_data_management() -> None:
    """Warn about secure runtime environments for data management."""
    if platform == "darwin":
        logger.warning(
            "LM Datasets MUST be stored only in market approved, well labeled, location(s) with "
            "shared access provided to other project members. "
            "Nothing critical should be stored in personal folders or laptops."
        )


def get_model_run_from_uri(uri: str) -> mlflow.entities.Run:
    """Get the model run based on a tracking server URI.

    Args:
        uri: The model URI. Currently supports the 'runs:/' and 'models:/' prefixes only.

    Returns:
        The Run.

    Raises:
        Exception: If the model URI is in an unsupported format.
    """
    if uri.startswith("runs:/"):
        # Grab the run id from the URI
        run_id = uri.strip("runs:/")
        run = mlflow.get_run(run_id)

        return run
    elif uri.startswith("models:/"):
        model_version_string = uri.split("/")

        mlflow_client = mlflow.MlflowClient()
        model_version = mlflow_client.get_model_version(model_version_string[1], model_version_string[2])

        run = mlflow.get_run(model_version.run_id)

        return run
    else:
        raise Exception(f"Unsupported URL: {uri}")

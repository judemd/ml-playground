import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import copy, copytree, ignore_patterns
from typing import Any, List, Dict

import dateutil.tz
import mlflow
import pandas as pd
from lit_ds_utils.decorate.logging import log_function
from lit_ds_utils.git_utils import repo_is_clean
from lit_ds_utils.mlflow_utils import (
    databricks_add_group_permissions_to_model_by_name,
    is_use_managed_mlflow,
)

from pygit2 import Repository

from .. import settings
from ..config.constants import (
    BASE_PATH,
    DATABRICKS_EXPERIMENT_NAME,
    DATABRICKS_GROUP_NAME,
    DATABRICKS_REGISTERED_MODEL_NAME
)
from ..deployment.model_wrapper import ModelWrapper
from ..features.feature_engineering import do_feature_engineering

EXPERIMENT_NAME = settings.str(DATABRICKS_EXPERIMENT_NAME)
MODEL_NAME = settings.str(DATABRICKS_REGISTERED_MODEL_NAME)
GROUP_NAME = settings.str(DATABRICKS_GROUP_NAME)

logger = logging.getLogger(__name__)


def wrap_and_log_model(model: Any, test_df: pd.DataFrame) -> None:
    """Wrap and log a model to MLFlow.

    Args:
        model: The model.
        test_df: Unprocessed test dataframe.
    """
    # Create the model wrapper
    wrapper = ModelWrapper()

    log_loss = -model.leaderboard(test_df)['score_test'][0]

    # Safety check - if running via `mlflow run` then the experiment must match what's in settings.ini
    _validate_experiment_ids()

    # Set experiment name
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Log the model metrics in mlflow.
        logger.info("Logging metrics with mlflow")
        _log_metrics(log_loss)

        # Log additional tags.
        logger.info("Logging tags with mlflow")
        _log_additional_tags()

        # Log artifacts.
        # logger.info("Logging artifacts with mlflow")
        # _log_artifacts()

        # Get the registered model name - optional
        model_name = None if str(MODEL_NAME).upper() == "NONE" or not MODEL_NAME.strip() else MODEL_NAME

        artifacts = {"predictor_path": model.path,
                     "target_encoder": 'target_encoder.pkl'}

        logger.info("Logging a model with MLFlow. Experiment name: %s. Model name: %s", EXPERIMENT_NAME, model_name)
        model_info = mlflow.pyfunc.log_model(
            registered_model_name=model_name,
            python_model=wrapper,
            artifact_path="model",
            artifacts=artifacts,
            code_path=_prepare_for_log_model(BASE_PATH),
            conda_env="conda.yaml",
            await_registration_for=10000
        )

        logger.info("Model run ID: %s logged.", model_info.run_id)

    if is_use_managed_mlflow() and model_name is not None:
        # Set CAN_MANAGE permissions on the registered model to the whole group
        logger.info("Allowing group %s permissions on model: %s", GROUP_NAME, MODEL_NAME)
        databricks_add_group_permissions_to_model_by_name(MODEL_NAME, GROUP_NAME)


def _validate_experiment_ids() -> None:
    """If running via mlflow run, ensure the --experiment-name argument has been provided, and matches the experiment name in settings.ini.

    Raises:
        Exception: If the experiment name is incorrect.
    """
    if "MLFLOW_EXPERIMENT_ID" in os.environ:
        mlflow_client = mlflow.MlflowClient()

        exp = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp and os.environ["MLFLOW_EXPERIMENT_ID"] != exp.experiment_id:
            logger.critical(
                "Experiments do not match. If running from `mlflow run`, specify the experiment name,"
                + f' `mlflow run . --experiment-name "{EXPERIMENT_NAME}" ...'
            )
            raise Exception("Mismatched experiment ids")


@log_function()
def _log_metrics(log_loss: float) -> None:
    """Log metrics on mlflow.

    Args:
        log_loss: Log loss
    """
    # mlflow.log_metric("accuracy", model_metrics["accuracy"])
    # mlflow.log_metric("precision", model_metrics["precision"])
    # mlflow.log_metric("recall", model_metrics["recall"])
    # mlflow.log_metric("f1", model_metrics["f1"])
    mlflow.log_metric("log_loss", log_loss)


@log_function()
def _log_artifacts() -> None:
    """Log artifacts on mlflow.

    Args:
    """
    mlflow.log_artifact('data/temp/artifacts/mlflow/feature_importance.csv', artifact_path="features")


@log_function()
def _log_additional_tags() -> None:
    """Log additional tags with the model.

    These tags represent the latest git commit id, timestamp and author, to aid traceability.
    """
    # Get git repo
    repo = Repository(BASE_PATH)
    last_commit = repo[repo.head.target]

    # Get current branch
    current_branch_short_name = repo.head.shorthand

    # Get the commit dt, and using offset convert it to eastern.
    tzinfo = timezone(timedelta(minutes=last_commit.author.offset))
    dt = datetime.fromtimestamp(float(last_commit.author.time), tzinfo)
    last_commit_timestamp_est = dt.astimezone(dateutil.tz.gettz("US/Eastern"))

    logger.debug("Last commit id: %s.", last_commit.short_id)
    logger.debug("Current branch: %s.", current_branch_short_name)
    logger.debug("Last commit timestamp in EST: %s.", last_commit_timestamp_est)
    logger.debug("Last commit author: %s", last_commit.author.name)

    # Create the tags against the run in mlflow
    mlflow.set_tag("last_commit_id", last_commit.short_id)
    mlflow.set_tag("current_branch", current_branch_short_name)
    mlflow.set_tag("last_commit_datetime_eastern", last_commit_timestamp_est)
    mlflow.set_tag("last_commit_author_name", last_commit.author.name)
    mlflow.set_tag("last_commit_author_email", last_commit.author.email)
    mlflow.set_tag("cloudforge_artifact_id", settings.str("CLOUD_FORGE_ARTIFACT_ID"))
    mlflow.set_tag("ds_model_pipeline_template_version", settings.str("CREATED_BY_GENERATOR_VERSION"))
    mlflow.set_tag("description", settings.str("MODEL_DESCRIPTION"))

    # If dirty workspace (un-committed files), then add an extra tag
    if not repo_is_clean(BASE_PATH):
        mlflow.set_tag("clean_workspace", "false")


def _prepare_for_log_model(code_dir: Path) -> List[str]:
    """Prepare the code within the supplied directory to be logged into managed MLFlow.  This will essentially make a \
       copy of the artifacts within the supplied directory, excluding any files which should not be logged, and return \
       a path to the copied location.

    Args:
        code_dir: The location of the code to be prepared.

    Returns:
        A list of path to the locations of the code which has had all the files which should not be logged to managed
        MLflow removed.
    """
    result_dir_path = Path(tempfile.TemporaryDirectory().name)

    # Copy pipeline directory across
    copytree(
        code_dir / "pipeline",
        result_dir_path / "pipeline",
        ignore=ignore_patterns("__pycache__", "**/temp/**", "tests", ".secrets.*", ".DS_Store", "logging.yml"),
    )

    # Copy settings.ini
    copy(code_dir / "settings.ini", result_dir_path / "settings.ini")

    # Copy run_pipeline.py
    copy(code_dir / "run_pipeline.py", result_dir_path / "run_pipeline.py")

    return [
        str(result_dir_path / "pipeline"),
        str(result_dir_path / "settings.ini"),
        str(result_dir_path / "run_pipeline.py"),
    ]

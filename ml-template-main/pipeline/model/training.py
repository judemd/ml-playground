"""Copyright (c) 2022, Liberty Mutual Group."""
import logging
from dataclasses import dataclass

import pandas as pd
from lit_ds_utils.decorate.logging import log_function
from sklearn.model_selection import train_test_split

from .. import settings
from ..config.constants import (
    DATABRICKS_EXPERIMENT_NAME,
    DATABRICKS_GROUP_NAME,
    DATABRICKS_REGISTERED_MODEL_NAME,
    MODEL_ARTIFACT,
)
from ..utils.utils import save_local_artifact
from .build import build_model
from .log_model import wrap_and_log_model

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = settings.str(DATABRICKS_EXPERIMENT_NAME)
MODEL_NAME = settings.str(DATABRICKS_REGISTERED_MODEL_NAME)
GROUP_NAME = settings.str(DATABRICKS_GROUP_NAME)


@dataclass()
class TrainTestSplits:
    """Train, test and holdout splits."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame


@log_function()
def train_and_log_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Run an ML Flow experiment and log to databricks using the args sent in.

    Args:
        train_df: Train df
        test_df: Test df.
    """
    logger.info("Building model")
    model = build_model(train_df=train_df)
    save_local_artifact(MODEL_ARTIFACT, model)

    logger.info("Logging model to MLFlow")
    wrap_and_log_model(model, test_df=test_df)


@log_function()
def get_train_test_splits(input_df: pd.DataFrame) -> TrainTestSplits:
    """Split the supplied data into training, test and holdout splits.

    Args:
        input_df: The dataframe to split.

    Returns:
        The train, test and holdout splits.
    """

    train_df = input_df[input_df['policy_year'] < 2017]
    test_df = input_df[input_df['policy_year'] == 2017]

    return TrainTestSplits(train_df, test_df)

"""Copyright (c) 2022, Liberty Mutual Group."""
import logging
from pathlib import Path

from lit_ds_utils.file_utils import delete_if_exists
from lit_ds_utils.mlflow_utils import mlflow_authenticate

from pipeline.acquisition.acquire_data import acquire_data
from pipeline.features.feature_engineering import do_feature_engineering
from pipeline.model.training import train_and_log_model, get_train_test_splits

logger = logging.getLogger(__name__)


@mlflow_authenticate
def run_pipeline() -> None:
    """Train and log the model to MLFlow."""

    # Get training data
    logger.info("Getting training data")
    input_data = acquire_data()
    input_data_copy = input_data.copy()  # Preserve a copy of the original input data

    logger.info("Getting train/test/holdout splits")
    train_test_splits = get_train_test_splits(input_data_copy)

    logger.info("Doing feature engineering")
    processed_train_df = do_feature_engineering(train_test_splits.train_df, training=True)
    processed_test_df = do_feature_engineering(train_test_splits.test_df, training=False)

    # Train model and log in mlflow
    train_and_log_model(train_df=processed_train_df,
                        test_df=processed_test_df)


def _clean() -> None:
    delete_if_exists(str(Path("/tmp") / "df-output.tab"))
    delete_if_exists(str(Path("/AutogluonModels")))


if __name__ == "__main__":
    _clean()
    logger.info("Running training...")
    run_pipeline()

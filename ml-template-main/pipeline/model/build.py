"""Copyright (c) 2022, Liberty Mutual Group."""
import logging
from typing import Any

import pandas as pd
from lit_ds_utils.decorate.logging import log_function
from autogluon.tabular import FeatureMetadata
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
import os

logger = logging.getLogger(__name__)


@log_function()
def build_model(train_df: pd.DataFrame, gpu: bool = False) -> Any:
    """Build the model.

    Args:
        train_df: DataFrame with features to train. Includes label.
        gpu: Whether there is a GPU to use multimodal

    Returns:
        The trained model object.
    """
    if gpu:
        predictor = MultiModalPredictor(label='target', eval_metric='log_loss', presets='medium_quality').fit(
            train_data=train_df,
            time_limit=10000,
        )
    else:
        feature_metadata = FeatureMetadata.from_df(train_df)
        hyperparameters = get_hyperparameter_config('extreme')
        predictor = TabularPredictor(label='target', eval_metric='log_loss', sample_weight='balance_weight').fit(
            train_data=train_df,
            # hyperparameters=hyperparameters,
            feature_metadata=feature_metadata,
            time_limit=10000,
            presets=['medium_quality', 'optimize_for_deployment'],
            feature_generator=AutoMLPipelineFeatureGenerator(vectorizer=TfidfVectorizer())
        )

    predictor_size = get_directory_size(predictor.path)

    logger.info(f"--------------- MODEL PATH IS SIZE {predictor_size} ---------------")

    return predictor


def get_directory_size(directory):
    total_size = 0
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)



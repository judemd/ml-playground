"""Copyright (c) 2022, Liberty Mutual Group."""
import logging


import pandas as pd
from lit_ds_utils.decorate.logging import log_function
from sklearn.pipeline import Pipeline
from pipeline.features.nlp_feature_engineering import NLPFeatureEngineering
from pipeline.features.categorical_feature_engineering import CategoricalFeatureEngineering
from pipeline.features.numerical_feature_engineering import NumericalFeatureEngineering
from pipeline.features.feature_engineering_utils import FeaturesToDrop

logger = logging.getLogger(__name__)


def drop_unused_features(df: pd.DataFrame, feature_list: list = FeaturesToDrop) -> pd.DataFrame:
    """
    Drops features that are not necessary for the modelling process after doing feature preprocessing

    Args:
        df (pd.DataFrame): Dataframe containing the features to drop.
        feature_list (list): A list of feature names to drop.

    Returns:
        df (pd.DataFrame): A Pandas DataFrame without the features found in feature_list.
    """
    return df.drop(labels=feature_list, axis=1)


@log_function()
def do_feature_engineering(input_data: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Example of feature engineering.

    Args:
        input_data: Input data.
        training: Called as part of model training?

    Returns:
        A Pandas DataFrame of the feature data.

    TODO: Implement this function and remove the reference implementation and implementation notes below.
    """
    logger.debug("****** feature engineering")
    final_fe_pipe = Pipeline(
        steps=[
            ("CategoricalFeatureEngineering", CategoricalFeatureEngineering(training)),
            ("NumericalFeatureEngineering", NumericalFeatureEngineering()),
            # ("NLPFeatureEngineering", NLPFeatureEngineering()),
        ]
    )
    if training:
        piped_data = final_fe_pipe.fit_transform(input_data)
    else:
        piped_data = final_fe_pipe.transform(input_data)

    input_data_dropped = drop_unused_features(piped_data, ['account_number', 'lob', 'split'])
    non_kw_cols = [c for c in input_data_dropped.columns if '_kw_' not in c]
    df_filtered = input_data_dropped[non_kw_cols]

    return df_filtered





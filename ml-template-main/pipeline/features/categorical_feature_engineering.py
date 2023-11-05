"""Copyright (c) 2022, Liberty Mutual Group.
Refer
Inspired by penguins/random forest tutorial here: https://datagy.io/sklearn-random-forests/
"""
import logging

import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import MEstimateEncoder
from pipeline.config.dataclasses import ModellingFeatures, NonModellingFeatures, CategoricalFeatures, TargetFeature
from pipeline.features.feature_engineering_utils import (
    IndustryGrouping,
    WeatherExposureGrouping,
    LitigationGrouping,
    CategoricalFeaturesToGroup,
    CategoricalFeaturesToOneHot,
    CategoricalFeaturesToTargetEncode
)

logger = logging.getLogger(__name__)

class CategoricalFeatureEngineering(BaseEstimator, TransformerMixin):
    """Performs Feature Engineering on categorical features"""

    def __init__(self, training):
        self.training = training

    @staticmethod
    def group_uncommon_categories(
        df: pd.DataFrame, feature: str, no_categories: int = 5
    ) -> pd.DataFrame:
        """
        Groups uncommon categorical values, only retaining the most frequently occurring values, rest are set
        to other.

        Args:
            df (pd.DataFrame): The pandas DataFrame to convert the loss state features in.
            feature (str): Feature in dataframe to perform the operation on.
            min_categories (int): Specifies the minimum amount of categories to keep, overriding top_percent.

        Returns:
            pd.DataFrame: The pandas DataFrame with the grouped loss state features.
        """
        # Get value counts for the column
        val_counts = df[feature].value_counts()
        # Get the top categories
        top_categories = val_counts.index[:no_categories]
        # Set the value of rows that are not in the top categories to "other"
        df[feature] = df[feature].apply(lambda x: x if x in top_categories else "other")
        logger.debug("Unique categories after grouping {} categories: {}".format(feature, df[feature].nunique()))
        return df

    @staticmethod
    def group_categories(
        df: pd.DataFrame, feature: str, grouped_feature: str, feature_grouping: dict) -> pd.DataFrame:
        """
        Groups categorical values based on a defined dictionary

        Args:
            df (pd.DataFrame): The pandas DataFrame to convert the loss state features in.
            feature (str): feature to perform the grouping on
            grouped_feature (str): the grouped feature that is returned
            feature_grouping (dict): Key-value pairs of features with their respective grouping

        Returns:
            pd.DataFrame: The pandas DataFrame with the grouped loss state features.
        """
        df[feature] = df[feature].astype('category')
        df[grouped_feature] = df[feature].replace(feature_grouping)
        return df


    @staticmethod
    def convert_categorical_dtypes(df: pd.DataFrame, feature_list: list = CategoricalFeatures) -> pd.DataFrame:
        """
        Converts a list of features to categorical dtype.

        Args:
            df (pd.DataFrame): Dataframe to perform the datatype conversions on.
            feature_list (list): A list of column names representing the categorical features to change into dtype category.

        Returns:
            df (pd.DataFrame): The dataset with proper dtypes.
        """
        if not df.empty:
            df[feature_list] = df[feature_list].astype("category")
        return df

    @staticmethod
    def one_hot_encode_categorical(df: pd.DataFrame, cols_to_encode: list):
        df = pd.get_dummies(df, columns = cols_to_encode)
        return df

    def target_encoding(self, df: pd.DataFrame, training: bool, cols_to_target_encode: list=CategoricalFeaturesToTargetEncode):
        if training:
            # m controls additive smoothing for regularisation, default = 1.0
            y = df[TargetFeature.target]
            target_encoder = MEstimateEncoder(cols=CategoricalFeaturesToTargetEncode, m=5.0)
            target_encoder.fit(df, y)
            transformed_df = target_encoder.transform(df)
            pickle.dump(target_encoder, open('target_encoder.pkl', 'wb'))
            return transformed_df
        else:
            target_encoder = pickle.load(open('target_encoder.pkl', 'rb'))
            transformed_df = target_encoder.transform(df)
            return transformed_df

    def fit(self, input_data, y=None):
        return self

    def transform(self, input_data: pd.DataFrame):
        logger.debug("Input data shape before categorical feature engineering {}".format(input_data.shape))
        input_data = self.convert_categorical_dtypes(input_data, CategoricalFeaturesToGroup)
        input_data = self.group_categories(input_data, ModellingFeatures.industry, 'industry_grouped', IndustryGrouping)
        input_data = self.group_categories(input_data, ModellingFeatures.state, 'litigation_grouped', LitigationGrouping)
        input_data = self.group_categories(input_data, ModellingFeatures.state, 'weather_grouped', WeatherExposureGrouping)
        input_data = self.group_categories(input_data, ModellingFeatures.state, 'GDP', GDP)
        input_data = self.group_categories(input_data, ModellingFeatures.state, 'LivingCost', LivingCost)
        input_data = self.group_categories(input_data, ModellingFeatures.state, 'CorporateTax', CorporateTax)
        input_data = self.group_categories(input_data, ModellingFeatures.state, 'AvgLaborCost', AvgLaborCost)
        input_data = self.target_encoding(df=input_data,
                                          training=self.training)
        input_data = self.one_hot_encode_categorical(input_data, cols_to_encode=CategoricalFeaturesToOneHot)
        logger.debug("Input data shape after categorical feature engineering {}".format(input_data.shape))
        return input_data

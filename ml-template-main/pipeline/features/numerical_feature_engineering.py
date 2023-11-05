"""Copyright (c) 2022, Liberty Mutual Group.

Inspired by penguins/random forest tutorial here: https://datagy.io/sklearn-random-forests/
"""
import logging

import numpy as np
import pandas as pd
from lit_ds_utils.decorate.logging import log_function
from sklearn.base import BaseEstimator, TransformerMixin
from pipeline.config.dataclasses import ModellingFeatures, NonModellingFeatures, NumericalFeatures

logger = logging.getLogger(__name__)


class NumericalFeatureEngineering(BaseEstimator, TransformerMixin):
    """Performs Feature Engineering on categorical features"""

    def fit(self, input_data, y=None):
        return self

    def transform(self, input_data: pd.DataFrame):
        logger.debug("Input data shape before numerical feature engineering {}".format(input_data.shape))
        input_data[ModellingFeatures.exposure_amt] = input_data[ModellingFeatures.exposure_amt]/1000
        input_data['state_industry_interact'] = input_data[ModellingFeatures.state] * input_data[ModellingFeatures.industry]
        input_data['state_has10k_interac'] = input_data[ModellingFeatures.state] * input_data[ModellingFeatures.has_10k]
        input_data['state_expsramt_interac'] = input_data[ModellingFeatures.state] * input_data[ModellingFeatures.exposure_amt]
        input_data['industry_has10k_interac'] = input_data[ModellingFeatures.industry] * input_data[ModellingFeatures.has_10k]
        input_data['industry_expsramt_interac'] = input_data[ModellingFeatures.industry] * input_data[ModellingFeatures.exposure_amt]
        input_data['expsramt_has10k_interac'] = input_data[ModellingFeatures.exposure_amt] * input_data[ModellingFeatures.has_10k]
        logger.debug("Input data shape after numerical feature engineering {}".format(input_data.shape))
        return input_data


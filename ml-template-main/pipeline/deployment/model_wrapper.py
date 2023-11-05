"""Copyright (c) 2022, Liberty Mutual Group."""
import logging
from typing import Any

import mlflow
import pandas as pd
from autogluon.tabular import TabularPredictor
from lit_ds_utils.decorate.logging import log_function

from ..features.feature_engineering import do_feature_engineering

logger = logging.getLogger(__name__)


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """MLFlow Python Model Wrapper."""

    def load_context(self, context):
        self.model = TabularPredictor.load(context.artifacts["predictor_path"])

    @log_function()
    def predict(self, context: Any, input_data: pd.DataFrame) -> pd.Series:
        """Use the packaged MLFlow model to make predictions.

        Reads in the input dataframe and returns the model predictions

        Args:
            context: An (Optional) :class:`~PythonModelContext` instance containing artifacts that the model can use to perform inference. It is not used in this model.
            input_data: Input data as a DataFrame.

        Returns:
            The model predictions.
        """
        processed_df = do_feature_engineering(input_data)
        logger.debug("processed_df shape {}".format(processed_df.shape))

        probability_scores = pd.Series(self.model.predict_proba(processed_df)[1])

        return probability_scores

    def predict_class(self, input_data: pd.DataFrame) -> pd.Series:
        processed_df = do_feature_engineering(input_data, training=False)
        logger.debug("processed_df shape {}".format(processed_df.shape))

        preds = pd.Series(self.model.predict(processed_df))

        return preds

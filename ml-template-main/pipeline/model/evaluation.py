"""Copyright (c) 2022, Liberty Mutual Group."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from lit_ds_utils.decorate.logging import log_function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from pipeline.deployment.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


@log_function()
def evaluate_model(model: ModelWrapper,
                   test_df: pd.DataFrame) -> dict:
    """Evaluate model and return dictionary of results

    Args:
        model (ModelWrapper): Model Wrapper.
        test_df (pd.DataFrame): Test dataframe.
    """
    y_test = test_df['target']
    X_test = test_df.drop(['target'], axis=1)

    y_prob = model.predict(input_data=X_test, context=None)
    y_pred = model.predict_class(input_data=X_test)

    # feature_importance = model.model.feature_importance(test_df)
    # feature_importance = feature_importance.reset_index().rename(columns={'index': 'variable'})
    # feature_importance.to_csv('data/temp/artifacts/mlflow/feature_importance.csv')

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    log_loss_val = log_loss(y_true=y_test, y_pred=y_prob, labels=[0.0, 1.0])

    return {"accuracy": accuracy, "precision": precision, "recall": recall, 'f1': f1, 'log_loss': log_loss_val}

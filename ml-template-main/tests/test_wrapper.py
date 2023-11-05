import pickle

import pandas as pd
import pytest

from pipeline.config.constants import MODEL_ARTIFACT
from pipeline.deployment.model_wrapper import ModelWrapper


# ======================================================================================================================
# Integration tests with model wrapper directly.
#
# These tests require:
#     1. The model to have been trained locally, via the run_pipeline.py script. This will create a 'model' file
#        under data/temp/artifacts/mlflow
#
# If the model file doesn't exists, the tests should be skipped.
# ======================================================================================================================

@pytest.mark.skipif(not MODEL_ARTIFACT.exists(), reason="model has not yet been trained")
def test_model_wrapper_predicts_value_successfully() -> None:
    # Load pickled model
    with open(MODEL_ARTIFACT, "rb") as model_file:
        # Load the pickled model
        model = pickle.load(model_file)
        # Create the wrapper around the pickled model
        wrapper = ModelWrapper(model)

        # Create a data frame with inference data
        test_data_df = pd.DataFrame({"example": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

        # Call predict on the model wrapper and get the output prediction
        output = wrapper.predict(context=None, input_data=test_data_df)
        assert output["prediction"] == "0.1"

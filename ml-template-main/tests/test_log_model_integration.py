import pickle
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from pipeline import settings
from pipeline.config.constants import (
    BASE_PATH,
    DATABRICKS_EXPERIMENT_NAME,
    DATABRICKS_REGISTERED_MODEL_NAME, MODEL_ARTIFACT,
)
from pipeline.deployment.model_wrapper import ModelWrapper
from pipeline.model.log_model import _prepare_for_log_model
from .test_utils import is_mlflow_server_up

# ======================================================================================================================
# Integration tests with local MLFlow instance
#
# These tests require:
#     1. The model to have been trained locally, via the run_pipeline.py script. This will create a 'model' file
#        under data/temp/artifacts/mlflow
#     2. A local mlflow server to have been started, on http://localhost:5000, via 'ds-cli start-mlflow' in a separate
#        terminal window.
#
# If the local mlflow server is not detected, or the model file doesn't exists, the tests should be skipped.
# ======================================================================================================================

@pytest.mark.skipif(not MODEL_ARTIFACT.exists(), reason="model has not yet been trained")
@pytest.mark.skipif(not is_mlflow_server_up(), reason="local mlflow has not been started")
@pytest.mark.skipif(settings.str("MLFLOW_TRACKING_URI") == "databricks", reason="configured to point to databricks")
def test_logged_model_from_mlflow_predicts_value_successfully(
    logged_model: Any
) -> None:
    # Call predict on the model wrapper and get the output prediction
    # Create a data frame with inference data
    test_data_df = pd.DataFrame({"example": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    # Call predict on the model wrapper and get the output prediction
    output = logged_model.predict(test_data_df)
    assert output["prediction"] == "0.1"


@pytest.fixture(scope="module")
def monkey_module(request):
    # Workaround to use monkeypatch in module scope
    # See: https://github.com/pytest-dev/pytest/issues/363
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def logged_model(monkey_module) -> Any:
    monkey_module.setenv("MLFLOW_TRACKING_URI", settings.str("MLFLOW_TRACKING_URI"))
    monkey_module.setenv("MLFLOW_API_CONTEXT_ROOT", "api/2.0/preview/mlflow")
    experiment_name = settings.str(DATABRICKS_EXPERIMENT_NAME) + "-test"
    model_name = settings.str(DATABRICKS_REGISTERED_MODEL_NAME) + "-test"

    # Load pickled model
    with open(MODEL_ARTIFACT, "rb") as model_file:
        # Load the pickled model
        model = pickle.load(model_file)
        # Create the wrapper around the pickled model
        wrapper = ModelWrapper(model)
        # Set the experiment name in MLFlow
        mlflow.set_experiment(experiment_name)
        try:
            with mlflow.start_run() as run:
                # Log the model.
                mlflow.pyfunc.log_model(
                    registered_model_name=model_name,
                    python_model=wrapper,
                    artifact_path="model",
                    pip_requirements=str(BASE_PATH / "requirements_generated.txt"),
                    code_path=_prepare_for_log_model(BASE_PATH),
                )

                # Load the model back out of mlflow again
                model_uri = f"runs:/{run.info.run_id}/model"
                loaded_model = mlflow.pyfunc.load_model(model_uri)

                # Yield the model to be used for inference
                yield loaded_model
        finally:
            client = MlflowClient()
            client.delete_run(run.info.run_id)
            client.delete_registered_model(name=model_name)

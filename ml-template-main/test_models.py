import pandas as pd
from dsc_2023_scoring_template import (
    configure_databricks,
    local_mlflow_prediction,
    validate_team_model
)
import os
os.environ["MLFLOW_CONDA_HOME"] = "/opt/cortex-installs/miniconda"

configure_databricks()

team_name: str = "im-on-smoko"
test_data_path: str = "s3://cortex-dsc-2023-data/sprint_data/sprint_train.parquet"
test_data = pd.read_parquet(test_data_path)

validate_team_model(
    team_name=team_name,
    test_data=test_data
)

model_preds = local_mlflow_prediction(
    model_uri='runs:/8d2e8370dab947f1946dfaf407a8fac4/model',
    data=test_data
)

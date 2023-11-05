"""Copyright (c) 2022, Liberty Mutual Group."""
import logging

import pandas as pd
from lit_ds_utils.decorate.logging import log_function

logger = logging.getLogger(__name__)


@log_function()
def acquire_data() -> pd.DataFrame:
    """Acquire the data.

    Returns:
        The dataset as a pandas DataFrame.
    """
    data = get_dataset("s3://cortex-dsc-2023-data/sprint_data/sprint_train.parquet")

    return data


def get_dataset(
    path_to_data: str,
) -> pd.DataFrame:
    """Load dataframe from disk

    Args:
        path_to_data (str): Absolute path to the data. Can be local or
            path to S3.
    """
    try:
        if path_to_data.endswith('.parquet'):
            # Load parquet file
            df = pd.read_parquet(path_to_data)
        elif path_to_data.endswith('.csv'):
            # Load CSV file
            df = pd.read_csv(path_to_data)
        elif path_to_data.endswith('.xlsx'):
            # Load Excel file
            df = pd.read_excel(path_to_data)
        elif path_to_data.endswith('.json'):
            # Load JSON file
            df = pd.read_json(path_to_data)
        return df
    except PermissionError as error:
        logger.error(f"You do not have access to the dataset in {path_to_data}")
        raise error

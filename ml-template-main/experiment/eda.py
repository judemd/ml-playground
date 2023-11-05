from pipeline.acquisition.acquire_data import acquire_data
from ydata_profiling import ProfileReport, compare
import pandas as pd


def convert_text_to_cat(df: pd.DataFrame, max_n: int = 100) -> pd.DataFrame:
    """Converts text variables to categorical based on level threshold.

    Args:
        df: Dataframe to convert.
        max_n: Max levels for variable to be considered categorical

    Returns:
        df: Dataframe with converted variables.
    """
    # loop over the columns in the dataframe
    for col in df.columns:
        # if the dtype of the column is object (i.e. text)
        if df[col].dtype == "object":
            # count the number of distinct values in the column
            n = len(df[col].unique())
            # if there are less than 10 distinct values
            if n <= max_n:
                # convert the dtype of the column to categorical
                df[col] = df[col].astype("category")

    return df


data = acquire_data()
data = convert_text_to_cat(data)

profile = ProfileReport(data, minimal=True)

profile.to_file("data_report.html")


target_1 = ProfileReport(data[data['target'] == 1], minimal=True)
target_0 = ProfileReport(data[data['target'] == 0], minimal=True)

comparison_report = compare([target_0, target_1])
comparison_report.to_file("comparison_report.html")

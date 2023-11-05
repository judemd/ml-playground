import shutil
import requests

from pipeline import settings

def is_mlflow_server_up() -> bool:
    """Is the local mlflow server up and listening?

    Returns:
        True if the server is listening on port 5000, and False otherwise.
    """
    try:
        mlflow_tracking_uri = settings.str("MLFLOW_TRACKING_URI")
        # Try to connect to local mlflow, but don't wait longer than a second.
        requests.get(mlflow_tracking_uri, verify=False, timeout=1)
        return True
    except Exception as e:
        return False


def is_conda_installed() -> bool:
    """Is the conda command installed?

    Returns:
        True if the conda command is installed, and False otherwise.
    """
    pipenv_path = shutil.which("conda")
    return pipenv_path is not None

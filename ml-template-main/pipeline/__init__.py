"""Top level application package."""
from pathlib import Path

from lit_ds_utils.settings import Settings

from .config.constants import BASE_PATH
from .config.logging_config import configure_logging

# Configure settings
files = [
    str(Path(BASE_PATH) / "settings.ini"),
    str(Path(BASE_PATH) / ".secrets.ini"),
]
settings = Settings(files=files)

# Configure logging
configure_logging()

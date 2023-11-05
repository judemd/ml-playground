import logging

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from pipeline.acquisition.acquire_data_utils import values_to_lower_case
from pipeline.config.dataclasses import ModellingFeatures
from pipeline.features.obfuscation_utils import (
    DEFAULT_MASKING_REGEX_PATTERNS,
    MaskTextCustomRegex,
    MaskTextSpacy,
)
from pipeline.utils.enums import FilePaths

logger = logging.getLogger(__name__)


def build_obfuscation_pipeline() -> Pipeline:
    text_obfuscation_pipeline = Pipeline(
        [
            (
                "1. Mask text with custom regex",
                MaskTextCustomRegex(custom_masking_regex_patterns=DEFAULT_MASKING_REGEX_PATTERNS),
            ),
            (
                "2. Mask text with spacy",
                MaskTextSpacy(spacy_model="en_core_web_md", entities_to_mask=["PERSON"]),
            ),
        ],
        verbose=True,
    )
    return text_obfuscation_pipeline


def obfuscate(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Obfuscate each record's main text note and subject
    Due to the large size of data, if split = True, the data will be split into
    smaller chunks so that it won't run into a memory error.
    """
    input_data[ModellingFeatures.FNOL_DESC] = input_data[ModellingFeatures.FNOL_DESC].astype(str)
    obfuscate_pipe = build_obfuscation_pipeline()
    logger.info("Obfuscating text")
    logger.debug(input_data[ModellingFeatures.FNOL_DESC].head(10))
    obfuscated_text = obfuscate_pipe.fit_transform(input_data[ModellingFeatures.FNOL_DESC])
    input_data[ModellingFeatures.FNOL_DESC] = obfuscated_text
    logger.info("Obfuscating text done")
    input_data = values_to_lower_case(input_data)
    logger.debug(input_data[ModellingFeatures.FNOL_DESC].head(10))
    # joblib.dump(obfuscate_pipe, FilePaths.OBFUSCATION_PIPELINE)
    # input_data.to_csv(FilePaths.OBFUSCATED_DATA_EXPOSUREONLY_PATH, index=False)
    # logger.info("Obfuscated data set at post-FNOI 30 day is saved.")
    return input_data

import logging
import unicodedata
import pandas as pd
from sklearn.base import TransformerMixin
from pipeline.features.nlp_feature_engineering_utils import (
    TEXT_CLEANING_FUNCTION_MAPPINGS,
    sw_to_remove,
    NLPFeatures,
)
logger = logging.getLogger(__name__)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NLPFeatureEngineering(TransformerMixin):
    def __init__(
        self,
        gsp_text_cleaning=TEXT_CLEANING_FUNCTION_MAPPINGS,
        features: list = NLPFeatures,
    ):
        self.features = features
        self.gsp_text_cleaning = gsp_text_cleaning

    @staticmethod
    def normalize_text(text: str) -> str:
        #  Normalizes text by decomposing accented, and special characters and ligatures.
        return unicodedata.normalize("NFKD", text)

    @staticmethod
    def calculate_sentiment(text) -> str:
        sentimentAnalyser = SentimentIntensityAnalyzer()
        scores = sentimentAnalyser.polarity_scores(text)
        # Extract the compound score
        compound_score = scores['compound']
        return compound_score

    def fit(self, input_data: pd.DataFrame, df_y=None):
        return self

    def transform(self, input_data: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            input_data[feature] = input_data[feature].astype(str)
            input_data[feature] = input_data[feature].str.lower()
            logger.info("Removing tabs and newlines")
            input_data[feature] = input_data[feature].replace(r"\n|\t", " ")
            logger.info("Normalizing text")
            input_data[feature] = input_data[feature].apply(self.normalize_text)
            for func_name, func in self.gsp_text_cleaning.items():
                logger.info(f"Starting feature engineering step: {func_name}")
                input_data[feature] = input_data[feature].apply(lambda x: func(x))
        input_data['1A_compound_score'] = input_data['item1A_summary'].apply(self.calculate_sentiment)
        input_data['3_compound_score'] = input_data['item3_summary'].apply(self.calculate_sentiment)
        input_data['7_compound_score'] = input_data['item7_summary'].apply(self.calculate_sentiment)
        input_data['7A_compound_score'] = input_data['item7A_summary'].apply(self.calculate_sentiment)

        return input_data

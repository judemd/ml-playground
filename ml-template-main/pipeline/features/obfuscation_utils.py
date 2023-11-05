import logging
import re
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import spacy
from sklearn.base import TransformerMixin
from spacy.tokens import Doc

# -----------------------------------------------------------------------------
# Values used for replacement in custom regex patterns
# -----------------------------------------------------------------------------
_POLICY_NUM_MASK = "<POLICY_NUM>"
_STATE_ZIP_MASK = "<STATE_ZIP>"
_PHONE_NUMBER_MASK = "<PH_NUM>"
_EMAIL_MASK = "<EMAIL>"
_ADDRESS_MASK = "<ADDRESS>"

# -----------------------------------------------------------------------------
# Custom NER Regex pattern
# -----------------------------------------------------------------------------
DEFAULT_MASKING_REGEX_PATTERNS: List[Tuple[str, str]] = [
    (r"H\d{2}\-?\d{3}\-?\d{6}\-?\d{2,3}", _POLICY_NUM_MASK),
    (
        r"policy number[\:\s]?\s?[A-Za-z\-]*\s?[0-9\-]+",
        f"Policy Number: {_POLICY_NUM_MASK}",
    ),
    (
        r"policy \#[\:\s]?\s?[A-Za-z\-]*\s?[0-9\-]+",
        f"Policy Number: {_POLICY_NUM_MASK}",
    ),
    (
        r"((A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY]))[\s\,][0-9]{5}(?:-[0-9]{4})?",
        _STATE_ZIP_MASK,
    ),
    (
        r"([a-zA-Z\#\@]*)[\(]?\d{3}[\)]?[^\w]?\d{3}[^\w]?\d{4}([^0-9])?",
        _PHONE_NUMBER_MASK,
    ),
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.?[a-zA-Z0-9-.]+", _EMAIL_MASK),
    (
        r"(PO Box)?(PO BOX)?\s?[0-9]*[A-Za-z\s\,]+[\s\,]+\<STATE_ZIP\>",
        _ADDRESS_MASK,
    ),
    (r"PO Box\s?[0-9]+[\s\,]*[A-Za-z]+[\s\,]*[A-Z]{2}", _ADDRESS_MASK),
    (r"PO BOX\s?[0-9]+[\s\,]*[A-Za-z]+[\s\,]*[A-Z]{2}", _ADDRESS_MASK),
]


class MaskTextSpacy(TransformerMixin):
    """
    Masked text to remove Personal Identifying Information
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_md",
        entities_to_mask: List[str] = None,
        spacy_exclude_components: List[str] = None,
        use_gpu: bool = False,
    ):
        """
        Args:
            spacy_model (str): Name of spacy model to use for entity
                masking.
            entities_to_mask (List[str]): List of Spacy entities to
                mask.
                Defaults to ["PERSON"].
                Other entities/options include:
                - Time indicator: ["TIME"], ["DATE"]
                - Money value: ["MONEY"]
                - Organization, companies, agencies, institutions: ["ORG"]
                - Geographical & geopolitical, countries, cities, states: ["GPE"]
                - Non-GPE locations, mountain ranges, bodies of water: ["LOC"]
                - Artifacts, Titles of books, songs, etc.: ["WORK_OF_ART"]
                - Named hurricanes, battles, wars, sports events, etc.: ["EVENT"]
                - Natural phenomenon: ["NAT"]
                - Buildings, airports, highways, bridges, etc.: ["FAC"]
                - Nationalities or religious or political groups ["NORP"]
                - Objects, vehicles, foods, etc. (Not services.): ["PRODUCT"]
                - Named documents made into laws: ["LAW"]
                - QUANTITY Measurements, as of weight or distance.
                - “first”, “second”, etc.: ["ORDINAL"]
                - Numerals that do not fall under another type: ["CARDINAL "]
                nlp = spacy.load("en_core_web_sm")
                ner_lst = nlp.pipe_labels["ner"]
            spacy_exclude_components (List[str]): List of spacy
                components to not download when loading the masking
                model. If you don’t need a particular component of the
                pipeline – for example, the tagger or the parser, you
                can disable or exclude it. This can sometimes make a
                big difference and improve loading and inference speed.
            cores (int): How many cores to run Spacy pipe with. If -1,
                uses all available cores on the machine.
        """
        self.spacy_model: str = spacy_model

        if entities_to_mask is None:
            entities_to_mask: List[str] = ["PERSON"]
        self.entities_to_mask: List[str] = entities_to_mask

        if spacy_exclude_components is None:
            spacy_exclude_components: List[str] = [
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ]
        self.spacy_exclude_components: List[str] = spacy_exclude_components

        self.use_gpu: bool = False

    def load_spacy_model(self):
        if self.use_gpu:
            spacy.require_gpu()  # Only use GPU if it is available
        else:
            spacy.require_cpu()

        return spacy.load(self.spacy_model, exclude=self.spacy_exclude_components)

    def _replace_entity(self, spacy_entity, text: str) -> str:
        """Replace entities identified in text with the entity type
        Replace the entity identified in the text with the spacy entity
        label if they are a type targeted for replacement.
        """
        if spacy_entity.label_ in self.entities_to_mask:
            start = spacy_entity.start_char
            end = start + len(spacy_entity.text)
            return text[:start] + "<" + spacy_entity.label_ + ">" + text[end:]
        return text

    def ner_substitution_with_spacy(self, spacy_doc: Doc) -> str:
        """Use spacy to replace NERs
        reversed to not modify the offsets of other entities when
        substituting
        """
        masked_loss_description = spacy_doc.text
        if masked_loss_description in ("nan", "NAN", "Nan", "NaN"):
            masked_loss_description = np.NaN
        else:
            for entity in reversed(spacy_doc.ents):
                masked_loss_description = self._replace_entity(entity, masked_loss_description)
        return masked_loss_description

    @staticmethod
    def convert_to_list_of_strings(text_to_mask: List[str]) -> List[str]:
        """Ensure that the text being masked is a list of strings"""
        logging.info("Convert the text to a list of strings")
        return [str(text) for text in text_to_mask]

    def mask_text(self, text_to_mask: List[str]) -> List[str]:
        """Mask the text using Spacy"""
        logging.info(f"Load Spacy model: {self.spacy_model}")
        ner_model = self.load_spacy_model()

        logging.info(
            f"Use Spacy to mask entities in the Loss Description "
            f"- Target Entities: {self.entities_to_mask} "
            f"- Number of lines: {len(text_to_mask)}"
        )
        # Create doc objects first to take advantage of spacy batch jobs
        logging.info("Create spacy document objects")
        spacy_docs: List[Doc] = list(
            ner_model.pipe(
                text_to_mask,
                disable=self.spacy_exclude_components,
                n_process=1,
            )
        )
        logging.info("Mask identified entities within the text using Spacy")
        spacy_ner_cleaned_text: List[str] = [
            self.ner_substitution_with_spacy(text_to_mask) for text_to_mask in spacy_docs
        ]
        return spacy_ner_cleaned_text

    def fit(self, text_array: pd.Series, df_y=None):
        return self

    def transform(self, text_array: Union[pd.Series, np.array]) -> np.array:
        return np.array(
            self.mask_text(self.convert_to_list_of_strings(text_array)),
            dtype="object",
        )


class MaskTextCustomRegex(TransformerMixin):
    """Masked text to remove Personal Identifying Information"""

    def __init__(
        self,
        custom_masking_regex_patterns: List[Tuple[str, str]] = None,
        ignore_case: bool = True,
    ):
        """
        Args:
            custom_masking_regex_patterns (List[Tuple[str, str]]): List
                of two element tuples, which contain regex patterns to
                search for within the string and values to replace the
                regex pattern with. The first element in the tuple is
                the regex pattern to search for. The second element is
                the value to use to replace the patterns within the
                text.
            ignore_case (bool): Should casing of text be ignored when matching
                regex patterns. Default: True.
        """
        if custom_masking_regex_patterns is None:
            custom_masking_regex_patterns = DEFAULT_MASKING_REGEX_PATTERNS
        self.custom_masking_regex_patterns: List[Tuple[str, str]] = custom_masking_regex_patterns
        self.ignore_case: bool = ignore_case

    def regex_ner_replacement(self, sentence: str):
        """Replace text given custom dictionary
        Go through a sentence and replace any tokens with patterns defined
        in the regex_mapping
        Args:
            sentence (str): String to check for regex patterns within.
        """
        if sentence:
            cleaned_sentence = sentence
            for (
                ner_regex_pattern,
                replacement_value,
            ) in self.custom_masking_regex_patterns:
                if self.ignore_case:
                    cleaned_sentence = re.sub(
                        ner_regex_pattern,
                        replacement_value,
                        cleaned_sentence,
                        flags=re.IGNORECASE,
                    )
                else:
                    cleaned_sentence = re.sub(ner_regex_pattern, replacement_value, cleaned_sentence)
            return cleaned_sentence
        return sentence

    def apply_custom_regex_patterns(self, text_to_clean: List[str]) -> List[str]:
        """Apply the custom regex patterns to clean the text
        Args:
            text_to_clean (List[str]): List of text strings to search for
                custom regex patterns within.
        """
        logging.info("Use custom regex patterns to mask the text")
        return [self.regex_ner_replacement(sentence) for sentence in text_to_clean]

    def fit(self, text_array: pd.Series, df_y=None):
        return self

    def transform(self, text_array: Union[pd.Series, np.array]) -> np.array:
        return np.array(self.apply_custom_regex_patterns(text_array))

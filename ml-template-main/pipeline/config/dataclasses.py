import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass()
class TargetFeature:
    target = "target"


@dataclass()
class ModellingFeatures:
    state = "state"
    industry = "industry"
    exposure_base = "exposure_base"
    exposure_amt = "exposure_amt"
    has_10k = "has_10k"



@dataclass()
class NonModellingFeatures:
    account_number = "account_number"
    policy_year = "policy_year"
    lob = "lob"
    split = "split"


@dataclass()
class CategoricalFeatures:
    state = "state"
    industry = "industry"
    exposure_base = "exposure_base"
    has_10k = "has_10k"


@dataclass()
class NumericalFeatures:
    exposure_amount = "exposure_amount"


@dataclass()
class TrainTestHoldoutSplit:
    """
    Train test holdout split
    """
    data_train: pd.DataFrame
    data_test: pd.DataFrame
    data_holdout: pd.DataFrame


@dataclass()
class SubSampledTrainSplit:
    """
    Train test holdout split
    """
    data_train: pd.DataFrame


@dataclass()
class TestHoldoutOpenSplit:
    """
    Train test holdout split
    """
    data_test: pd.DataFrame
    data_holdout: pd.DataFrame


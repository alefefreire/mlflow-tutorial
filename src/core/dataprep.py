from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd  # type: ignore
from mlflow.entities import Experiment


class DataPrep(ABC):
    """
    Abstract class for data preparation.
    This class is responsible for preparing the dataset for training and testing.
    It should be inherited by specific data preparation classes for different datasets.
    Attributes
    ----------
    dataset : pd.DataFrame
        The dataset to be prepared.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    @abstractmethod
    def get_X_and_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the features and target from the dataset.
        """
        pass


class DataSplitter(ABC):
    """
    Abstract class for data splitting.
    This class is responsible for splitting the dataset into training and testing sets.
    It should be inherited by specific data splitting classes for different datasets.
    Attributes
    ----------
    experiment : Experiment
        The experiment object to be logged using MLflow.
    """

    def __init__(self, experiment: Experiment = None):
        self._experiment = experiment

    @abstractmethod
    def split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and testing sets.
        """
        pass

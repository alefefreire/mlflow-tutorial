from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.classifier import ClassifierModel


class ModelTrainer(ABC):
    """
    Abstract class for model training.
    This class defines the interface for model training
    """

    @abstractmethod
    def create_pipeline(self) -> Pipeline:
        """
        Create a model pipeline.
        """
        pass

    @abstractmethod
    def cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model: ClassifierModel,
        n_splits: int,
        random_state: int,
    ):
        """
        Perform cross-validation on the model.
        """
        pass

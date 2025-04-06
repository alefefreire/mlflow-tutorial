from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator

from notebooks.src.models.classifier import ClassifierModel
from notebooks.src.models.regressor import RegressorModel


class BaseMLPipeline(ABC):
    """
    Base class for all ML pipelines.
    """

    @abstractmethod
    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RegressorModel | ClassifierModel:
        """
        Train the model.
        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        y_train : pd.Series
            The target variable.
        Returns
        -------
        RegressorModel | ClassifierModel
            The trained model.

        """
        pass

    @abstractmethod
    def tune_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, baseline: BaseEstimator
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Tune the model hyperparameters.
        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        y_train : pd.Series
            The target variable.
        baseline : BaseEstimator
            The baseline model to tune.
        Returns
        -------
        BaseEstimator
            The tuned sklearn model.
        """
        pass

    @abstractmethod
    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model: ClassifierModel | RegressorModel,
    ) -> None:
        """
        This method evaluates the model
        Parameters
        ----------
        X_test : pd.DataFrame
            The test data.
        y_test : pd.Series
            The target variable.
        model : ClassifierModel | RegressorModel
            The trained model.
        Returns
        -------
        None
        """
        pass

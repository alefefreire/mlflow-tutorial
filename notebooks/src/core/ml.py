from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator


class BaseMLPipeline(ABC):
    """
    Base class for all ML pipelines.
    """

    @abstractmethod
    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, model: BaseEstimator
    ) -> BaseEstimator:
        """
        Train the model.
        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        y_train : pd.Series
            The target variable.
        model : BaseEstimator
            The sklearn model to train.
        Returns
        -------
        BaseEstimator
            The trained sklearn model.
        """
        pass

    @abstractmethod
    def compare_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseEstimator:
        """
        Compare different models and return the best one.
        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        y_train : pd.Series
            The target variable.
        Returns
        -------
        BaseEstimator
            The sklearn model with the best performance.
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

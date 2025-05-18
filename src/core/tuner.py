from abc import ABC, abstractmethod

import pandas as pd
from mlflow.entities import Experiment

from src.models.classifier import ClassifierModel
from src.models.params import Params
from src.models.regressor import RegressorModel


class ModelTuner(ABC):
    """Abstract base class for model tuning strategies.

    This class defines the interface for different model tuning approaches,
    such as hyperparameter search using cross-validation. Subclasses must
    implement the `create_search_cv` method to provide a specific tuning strategy.
    """

    def __init__(self, tuner_params: Params, experiment: Experiment):
        self.tuner_params = tuner_params
        self._experiment = experiment

    @abstractmethod
    def create_search_cv(self) -> ClassifierModel | RegressorModel:
        """
        This the abstract method for search cv strategy,
        should be inherited and ovewrited by the user.
        """
        pass

    @abstractmethod
    def tune(
        self,
        X_train: pd.DataFrame,
        y_traine: pd.Series,
        baseline: ClassifierModel | RegressorModel,
    ) -> ClassifierModel | RegressorModel:
        """
        Abstract method for tuning a model using the provided training data and baseline model.

        This method should be implemented by subclasses to define the specific tuning strategy,
        such as hyperparameter optimization or model refinement.

        Parameters
        ----------
        X_train : pd.DataFrame
            The feature matrix for training the model.
        y_train : pd.Series
            The target variable for training the model.
        baseline : ClassifierModel | RegressorModel
            The baseline model to compare against during the tuning process.

        Returns
        -------
        ClassifierModel | RegressorModel
            The tuned model after applying the tuning strategy.

        Notes
        -----
        - Subclasses must override this method to provide a concrete implementation.
        - The tuning process may involve techniques such as grid search, random search, or Bayesian optimization.
        """
        pass

    @abstractmethod
    def nested_cv(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """
        Abstract method for doing nested cross-validation (CV) strategy

        This method should be implemented by subclasses to define the specific nested cv strategy.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        y_train : pd.Series
            The target variable.
        baseline : ClassifierModel | RegressorModel
            The baseline model to tune.
        Returns
        -------
        None
        """
        pass

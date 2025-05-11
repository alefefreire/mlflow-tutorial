from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.classifier import ClassifierModel
from src.models.regressor import RegressorModel


class ModelTrainer(ABC):
    """
    Creates a machine learning pipeline.

    This method should be implemented by subclasses to define the specific steps
    of the pipeline, such as preprocessing, feature selection, and model training.

    Returns
    -------
    Pipeline
        A scikit-learn Pipeline object containing the defined steps.
    """

    @abstractmethod
    def create_pipeline(self) -> Pipeline:
        """
        Create the pipeline.
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
    ) -> ClassifierModel | RegressorModel:
        """
        Performs cross-validation on the given model.

        This method should be implemented by subclasses to define the specific
        cross-validation strategy, including splitting the data, training the model,
        and evaluating its performance.

        Parameters
        ----------
        X_train : pd.DataFrame
            The feature matrix for training the model.
        y_train : pd.Series
            The target variable for training the model.
        X_test : pd.DataFrame
            The feature matrix for testing the model.
        y_test : pd.Series
            The target variable for testing the model.
        model : ClassifierModel
            The machine learning model to be evaluated.
        n_splits : int
            The number of splits for cross-validation.
        random_state : int
            The random seed for reproducibility.

        Returns
        -------
        ClassifierModel | RegressorModel
            The trained model after cross-validation, wrapped in a `ClassifierModel`
            or `RegressorModel` object.
        """
        pass

from typing import Tuple

import mlflow
import pandas as pd
from mlflow.entities import Experiment
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.core.dataprep import DataPrep, DataSplitter
from src.core.ml import BaseMLPipeline
from src.core.trainer import ModelTrainer
from src.core.tuner import ModelTuner
from src.models.classifier import ClassifierModel
from src.models.regressor import RegressorModel


class MLPipeline(BaseMLPipeline):
    """
    This class implements the ML pipeline interface.
    It is responsible for orchestrating the data preparation, data splitting,
    model training, and evaluation processes.
    It uses the abstract classes defined in the core module to ensure that
    the pipeline can be extended for different datasets and models.
    Attributes
    ----------
    dataset : Dataset
        The dataset to be used in the pipeline.
    data_prep : DataPrep
        The data preparation object to be used in the pipeline.
    data_splitter : DataSplitter
        The data splitting object to be used in the pipeline.
    model_trainer : ModelTrainer
        The model training object to be used in the pipeline.
    experiment : Experiment
        The experiment object to be logged using MLflow.
    """

    _type_constraints = {
        "data_prep": DataPrep,
        "data_splitter": DataSplitter,
        "model_trainer": ModelTrainer,
        "model_tuner": ModelTuner,
    }

    def __init__(
        self,
        data_prep: DataPrep,
        data_splitter: DataSplitter,
        model_trainer: ModelTrainer,
        model_tuner: ModelTuner,
        experiment: Experiment,
    ):
        self._experiment = experiment
        self._data_prep = data_prep
        self._data_splitter = data_splitter
        self._model_trainer = model_trainer
        self._model_tuner = model_tuner

        # Initiate the typed attributes
        for name, value in {
            "data_prep": data_prep,
            "data_splitter": data_splitter,
            "model_trainer": model_trainer,
            "model_tuner": model_tuner,
        }.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        if name in self._type_constraints:
            expected_type = self._type_constraints[name]
            private_name = f"_{name}"
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} must be an instance of {expected_type.__name__}"
                )
            super().__setattr__(private_name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._type_constraints:
            private_name = f"_{name}"
            try:
                return super().__getattribute__(private_name)
            except AttributeError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def train_test_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Interface to split the dataset into train and test sets.
        """

        X, y = self._data_prep.get_X_and_y()
        return self._data_splitter.split_data(X, y)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ClassifierModel:
        """
        Interface for model training
        """

        trained_model = self._model_trainer.cross_validate(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        return trained_model

    def tune_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        baseline: ClassifierModel | RegressorModel,
    ) -> ClassifierModel | RegressorModel:
        """
        Tune the model hyperparameters.
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
        ClassifierModel | RegressorModel
            The tuned sklearn model.
        """

        return self._model_tuner.tune(
            X_train=X_train, y_train=y_train, baseline=baseline
        )

    def nested_cv_(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """
        Tune the model hyperparameters.
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

        return self._model_tuner.nested_cv(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

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
        """
        model: Pipeline = model.model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        if self._experiment is not None:
            with mlflow.start_run(
                experiment_id=self._experiment.experiment_id,
                run_name="evaluate_model",
            ):
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc_auc", roc_auc)

    def run_pipeline(
        self,
    ) -> None:
        """
        Run the pipeline.
        """

        X_train, X_test, y_train, y_test = self.train_test_split()

        baseline = self.train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        _ = self.tune_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            baseline=baseline,
        )
        _ = self.nested_cv_(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

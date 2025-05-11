import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Experiment
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.core.trainer import ModelTrainer
from src.models.classifier import ClassifierModel
from src.models.params import Estimators, Params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.custom.trainer")


class CustomModelTrainer(ModelTrainer):
    """
    A custom implementation of the ModelTrainer class for training machine learning models.

    This class is designed to handle the creation of a machine learning pipeline,
    perform cross-validation, and log metrics to MLflow for experiment tracking.
    It supports binary classification tasks and integrates preprocessing steps
    and model training into a unified pipeline.

    Attributes
    ----------
    pre_processing : Params
        Parameters for the preprocessing step in the pipeline.
    estimators : Estimators
        The machine learning models or estimators to be used in the pipeline.
    experiment : Experiment
        An MLflow Experiment object for logging metrics and tracking experiments.
    n_splits : int
        The number of splits for StratifiedKFold cross-validation.
    random_state : int
        The random seed for reproducibility during cross-validation.
    Methods
    -------
    create_pipeline() -> Pipeline
        Constructs a machine learning pipeline with preprocessing and model steps.
    cross_validate(X_train, y_train, X_test, y_test, n_splits, random_state) -> ClassifierModel
        Performs StratifiedKFold cross-validation, calculates performance metrics,
        logs metrics to MLflow, and returns a trained model.
    """

    def __init__(
        self,
        pre_processing: Params,
        estimators: Estimators,
        experiment: Experiment,
        n_splits: int,
        random_state: int,
    ):
        self._experiment = experiment
        self.pre_processing = pre_processing
        self.estimators = estimators
        self.n_splits = n_splits
        self.random_state = random_state

    def create_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        """
        Constructs a machine learning pipeline with preprocessing and model steps.

        Parameters
        ----------
        estimator : BaseEstimator
            The estimator to be used in the pipeline.

        Returns
        -------
        Pipeline
            A scikit-learn Pipeline object containing the preprocessing and model steps.
        """
        return Pipeline(
            steps=[
                (
                    "pre_processing",
                    self.pre_processing.method,
                ),
                ("model", estimator),
            ]
        )

    def cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ClassifierModel:
        """
        Trains a binary classification model using StratifiedKFold cross-validation.

        This method builds an XGBoost classifier with a power transformer preprocessing step,
        performs k-fold cross-validation, calculates performance metrics for each fold,
        logs the average metrics, and generates ROC curve visualizations.

        Parameters
        ----------
        X_train : pd.DataFrame
            The feature matrix for training the model.
        y_train : pd.Series
            The target variable (binary class labels).
        X_test: pd.DataFrame
            The feature matrix for testing the model.
        y_test: pd.Series
            The target variable (binary class labels) for testing.

        Returns
        -------
        ClassifierModel | RegressorModel
            A model wrapper object containing the trained pipeline, parameters,
            and performance score. Though the return type suggests it could be
            either a classifier or regressor, the implementation currently
            returns a ClassifierModel.

        Notes
        -----
        - The method builds a pipeline with a PowerTransformer and XGBClassifier.
        - Cross-validation metrics calculated include: accuracy, precision, recall, F1, and AUC-ROC.
        - Generates and logs a ROC curve visualization showing performance across all folds.
        - All metrics are logged to MLflow for experiment tracking.
        """
        all_acc = []
        all_pr = []
        all_recall = []
        all_f1 = []
        all_auc = []

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        trained_models = dict()
        for model_name, estimator in self.estimators.model_dump().items():
            for fold, (train, test) in enumerate(skf.split(X_train, y_train)):
                X_train_fold, X_test_fold = (
                    X_train.iloc[train],
                    X_train.iloc[test],
                )
                y_train_fold, y_test_fold = (
                    y_train.iloc[train],
                    y_train.iloc[test],
                )
                model = self.create_pipeline(estimator=estimator)

                # Fit the model
                model.fit(X_train_fold, y_train_fold)
                y_pred_fold = model.predict(X_test_fold)

                # Score the model
                score = f1_score(y_test_fold, y_pred_fold, average="weighted")
                acc = accuracy_score(y_test_fold, y_pred_fold)
                precision = precision_score(
                    y_test_fold, y_pred_fold, average="weighted"
                )
                recall = recall_score(y_test_fold, y_pred_fold, average="weighted")
                roc_auc = roc_auc_score(y_test_fold, y_pred_fold)

                all_f1.append(score)
                all_acc.append(acc)
                all_pr.append(precision)
                all_recall.append(recall)
                all_auc.append(roc_auc)

                logger.info(
                    f"Fold {fold + 1}/{self.n_splits} - F1: {score:.4f}, AUC: {roc_auc:.4f}"
                )
                logger.info(f"Fold {fold + 1}/{self.n_splits} - Accuracy: {acc:.4f}")
                logger.info(
                    f"Fold {fold + 1}/{self.n_splits} - Precision: {precision:.4f}"
                )
                logger.info(f"Fold {fold + 1}/{self.n_splits} - Recall: {recall:.4f}")
                logger.info(f"Fold {fold + 1}/{self.n_splits} - ROC AUC: {roc_auc:.4f}")

            avg_acc = np.mean(all_acc)
            avg_pr = np.mean(all_pr)
            avg_recall = np.mean(all_recall)
            avg_f1 = np.mean(all_f1)
            avg_auc = np.mean(all_auc)
            logger.info(f"\nModel: {model_name} - Average Metrics:")
            logger.info(f"Average Accuracy: {avg_acc}")
            logger.info(f"Average Precision: {avg_pr}")
            logger.info(f"Average Recall: {avg_recall}")
            logger.info(f"Average F1: {avg_f1}")
            logger.info(f"Average AUC: {avg_auc}")

            # Train on all data
            final_model = self.create_pipeline(estimator)
            final_model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred_test = final_model.predict(X_test)
            test_score = f1_score(y_test, y_pred_test, average="weighted")

            if self._experiment is not None:
                with mlflow.start_run(
                    experiment_id=self._experiment.experiment_id,
                    run_name=f"train_model_{model_name}",
                ):
                    mlflow.log_metric("avg_accuracy", avg_acc)
                    mlflow.log_metric("avg_precision", avg_pr)
                    mlflow.log_metric("avg_recall", avg_recall)
                    mlflow.log_metric("avg_f1", avg_f1)
                    mlflow.log_metric("avg_auc", avg_auc)

            trained_models[model_name] = ClassifierModel(
                name=final_model.__class__.__name__,
                model=final_model,
                params=Params(**final_model.get_params()),
                score=test_score,
            )
        best_model = max(trained_models.items(), key=lambda x: x[1].score)
        return best_model[1]

import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Experiment
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier

from src.core.trainer import ModelTrainer
from src.models.classifier import ClassifierModel
from src.models.params import Params
from src.services.plots import plot_cross_validated_roc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.custom.trainer")


class CustomModelTrainer(ModelTrainer):
    def __init__(self, experiment: Experiment = None):
        self._experiment = experiment

    def create_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "power_transformer",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=100,
                        objective="binary:logistic",
                        random_state=42,
                    ),
                ),
            ]
        )

    def cross_validate(
        self,
        model: ClassifierModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
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
        n_splits : int, default=5
            Number of folds for k-fold cross-validation.
        random_state : int, default=42
            Random seed for reproducibility in StratifiedKFold splitting.

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
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        for fold, (train, test) in enumerate(skf.split(X_train, y_train)):
            X_train_fold, X_test_fold = (
                X_train.iloc[train],
                X_train.iloc[test],
            )
            y_train_fold, y_test_fold = (
                y_train.iloc[train],
                y_train.iloc[test],
            )

            # Fit the model
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)

            # Score the model
            score = f1_score(y_test_fold, y_pred_fold, average="weighted")
            acc = accuracy_score(y_test_fold, y_pred_fold)
            precision = precision_score(y_test_fold, y_pred_fold, average="weighted")
            recall = recall_score(y_test_fold, y_pred_fold, average="weighted")
            roc_auc = roc_auc_score(y_test_fold, y_pred_fold)

            all_f1.append(score)
            all_acc.append(acc)
            all_pr.append(precision)
            all_recall.append(recall)
            all_auc.append(roc_auc)

            logger.info(
                f"Fold {fold + 1}/{n_splits} - F1: {score:.4f}, AUC: {roc_auc:.4f}"
            )
            logger.info(f"Fold {fold + 1}/{n_splits} - Accuracy: {acc:.4f}")
            logger.info(f"Fold {fold + 1}/{n_splits} - Precision: {precision:.4f}")
            logger.info(f"Fold {fold + 1}/{n_splits} - Recall: {recall:.4f}")
            logger.info(f"Fold {fold + 1}/{n_splits} - ROC AUC: {roc_auc:.4f}")

        avg_acc = np.mean(all_acc)
        avg_pr = np.mean(all_pr)
        avg_recall = np.mean(all_recall)
        avg_f1 = np.mean(all_f1)
        avg_auc = np.mean(all_auc)

        logger.info(f"Average Accuracy: {avg_acc}")
        logger.info(f"Average Precision: {avg_pr}")
        logger.info(f"Average Recall: {avg_recall}")
        logger.info(f"Average F1: {avg_f1}")
        logger.info(f"Average AUC: {avg_auc}")

        # Train in all data
        _ = model.fit(X_train, y_train)
        # Plot ROC curve
        roc_fig, _, _ = plot_cross_validated_roc(
            X=X_train,
            y=y_train,
            classifier=model,
            n_splits=n_splits,
            random_state=random_state,
        )

        if self._experiment is not None:
            with mlflow.start_run(
                experiment_id=self._experiment.experiment_id,
                run_name="train_model",
            ):
                mlflow.log_metric("avg_accuracy", avg_acc)
                mlflow.log_metric("avg_precision", avg_pr)
                mlflow.log_metric("avg_recall", avg_recall)
                mlflow.log_metric("avg_f1", avg_f1)
                mlflow.log_metric("avg_auc", avg_auc)
                mlflow.log_figure(roc_fig, "roc_curve.png")

        return ClassifierModel(
            name=model.__class__.__name__,
            model=model,
            params=Params(**model.get_params()),
            score=score,
        )

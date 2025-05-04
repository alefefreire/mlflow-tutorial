import logging
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Experiment
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from src.core.components import DataPrep, DataSplitter
from src.core.ml import BaseMLPipeline
from src.models.classifier import ClassifierModel
from src.models.data import Dataset
from src.models.params import Params
from src.models.regressor import RegressorModel
from src.services.plots import plot_classification_report, plot_cross_validated_roc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.services.pipeline")


class MLPipeline(BaseMLPipeline):

    def __init__(self, dataset: Dataset, experiment: Experiment = None):
        self._dataset = dataset
        self._experiment = experiment
        self._data_prep = DataPrep(dataset)
        self._data_splitter = DataSplitter(experiment)

    def train_test_split(
        self,
        test_size: float = 0.2,
        is_stratified: bool = False,
        selected_features: List[str] = None,
        is_drop_id: bool = True,
        feature_Id: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Interface to split the dataset into train and test sets.
        """
        logger.info("Splitting the dataset into train and test sets.")

        X, y = self._data_prep.get_features_and_target(
            selected_features=selected_features,
            is_drop_id=is_drop_id,
            feature_Id=feature_Id,
        )
        return self._data_splitter.split_data(
            X,
            y,
            test_size=test_size,
            is_stratified=is_stratified,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> ClassifierModel | RegressorModel:
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
        model = Pipeline(
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

    def tune_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        baseline: ClassifierModel | RegressorModel,
        n_iter: int,
        cv: int,
        n_jobs: int,
        verbose: int,
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
        model: Pipeline = baseline.model
        params = Params(
            model__max_depth=Integer(3, 10),
            model__learning_rate=Real(0.01, 0.3, prior="log-uniform"),
            model__subsample=Real(0.5, 1.0),
            model__colsample_bytree=Real(0.5, 1.0),
        )
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=params.model_dump(),
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring="roc_auc",
        )
        _ = bayes_search.fit(X_train, y_train)
        if self._experiment is not None:
            with mlflow.start_run(
                experiment_id=self._experiment.experiment_id,
                run_name="tune_model",
            ):
                mlflow.log_param("best_params", bayes_search.best_params_)
                mlflow.log_metric("roc_auc", bayes_search.best_score_)
                mlflow.sklearn.log_model(
                    sk_model=bayes_search.best_estimator_,
                    artifact_path="tuned_model",
                )
        logger.info(f"Best params: {bayes_search.best_params_}")
        logger.info(f"Best roc auc score: {bayes_search.best_score_}")
        logger.info(f"Best estimator: {bayes_search.best_estimator_}")
        logger.info(
            f"Best estimator name: {bayes_search.best_estimator_._final_estimator.__class__.__name__}"
        )
        logger.info(
            f"Best estimator params: {bayes_search.best_estimator_._final_estimator.get_params()}"
        )
        if baseline.score >= bayes_search.best_score_:
            logger.info(
                f"Baseline model score: {baseline.score} is better than tuned model score: {bayes_search.best_score_}"
            )
            return baseline
        else:
            logger.info(
                f"Tuned model score: {bayes_search.best_score_} is better than baseline model score: {baseline.score}"
            )
            return ClassifierModel(
                name=bayes_search.best_estimator_._final_estimator.__class__.__name__,
                model=bayes_search.best_estimator_,
                params=bayes_search.best_params_,
                score=bayes_search.best_score_,
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

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1: {f1}")
        logger.info(f"ROC AUC: {roc_auc}")

        cm_display = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            cmap="Blues",
            normalize=None,
        )

        clf_report_fig = plot_classification_report(
            y_true=y_test,
            y_pred=y_pred,
            figsize=(10, 6),
            output_dict=True,
        )
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
                mlflow.log_figure(cm_display.figure_, "confusion_matrix.png")
                mlflow.log_figure(clf_report_fig, "classification_report.png")

    def run_pipeline(self) -> None:
        """
        Run the pipeline.
        """

        X_train, X_test, y_train, y_test = self.train_test_split(
            is_drop_id=True,
            is_stratified=True,
        )

        baseline = self.train(X_train, y_train)
        tuned_model = self.tune_model(
            X_train,
            y_train,
            baseline=baseline,
            n_iter=10,
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        _ = self.evaluate_model(X_test, y_test, tuned_model)
        logger.info("Pipeline run completed.")

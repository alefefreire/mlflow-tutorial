import logging
from typing import List, Tuple

import mlflow
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
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from src.core.dataprep import DataPrep, DataSplitter
from src.core.ml import BaseMLPipeline
from src.core.trainer import ModelTrainer
from src.models.classifier import ClassifierModel
from src.models.data import Dataset
from src.models.params import Params
from src.models.regressor import RegressorModel
from src.services.plots import plot_classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.services.pipeline")


class MLPipeline(BaseMLPipeline):

    def __init__(
        self,
        dataset: Dataset,
        experiment: Experiment = None,
        data_prep: DataPrep = None,
        data_splitter: DataSplitter = None,
        model_trainer: ModelTrainer = None,
    ):
        self._dataset = dataset
        self._experiment = experiment
        self._data_prep = data_prep
        self._data_splitter = data_splitter
        self._model_trainer = model_trainer

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

        X, y = self._data_prep.get_X_and_y(
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
    ) -> ClassifierModel:
        """
        Interface for model training
        """
        model_pipeline = self._model_trainer.create_pipeline()
        trained_model = self._model_trainer.cross_validate(
            X_train=X_train,
            y_train=y_train,
            model=model_pipeline,
            n_splits=n_splits,
            random_state=random_state,
        )

        return trained_model

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

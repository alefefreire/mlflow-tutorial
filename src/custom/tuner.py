import logging

import pandas as pd
from mlflow.entities import Experiment

from src.core.gridsearch import MLflowGridSearchCV
from src.core.tuner import ModelTuner
from src.models.classifier import ClassifierModel
from src.models.params import Params
from src.models.regressor import RegressorModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.services.tuner")


class CustomModelTuner(ModelTuner):
    """
    CustomModelTuner is a class that extends the ModelTuner to provide
    specific tuning capabilities for machine learning models. It utilizes
    parameters defined in the Params class and integrates with MLflow
    for experiment tracking.
    """

    def __init__(self, tuner_params: Params, experiment: Experiment):
        self.tuner_params = tuner_params
        self._experiment = experiment

    def create_search_cv(self) -> MLflowGridSearchCV:

        return MLflowGridSearchCV(
            experiment=self._experiment,
            enable_mlflow=self.tuner_params.enable_mlflow,
            estimator=self.tuner_params.pipeline,
            param_grid=[param.model_dump() for param in self.tuner_params.param_grid],
            scoring=self.tuner_params.scoring,
            cv=self.tuner_params.cv,
            n_jobs=self.tuner_params.n_jobs,
            verbose=self.tuner_params.verbose,
        )

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        baseline: ClassifierModel | RegressorModel,
    ) -> ClassifierModel | RegressorModel:

        gscv = self.create_search_cv()
        gscv.fit(X_train, y_train)
        logger.info(f"Best params: {gscv.best_params_}")
        logger.info(f"Best roc auc score: {gscv.best_score_}")
        logger.info(f"Best estimator: {gscv.best_estimator_}")
        logger.info(
            f"Best estimator name: {gscv.best_estimator_._final_estimator.estimator.__class__.__name__}"
        )
        logger.info(
            f"Best estimator params: {gscv.best_estimator_._final_estimator.get_params()}"
        )
        if baseline.score >= gscv.best_score_:
            logger.info(
                f"Baseline model score: {baseline.score} is better than tuned model score: {gscv.best_score_}"
            )
            return baseline
        else:
            logger.info(
                f"Tuned model score: {gscv.best_score_} is better than baseline model score: {baseline.score}"
            )
            return ClassifierModel(
                name=gscv.best_estimator_._final_estimator.__class__.__name__,
                model=gscv.best_estimator_,
                params=gscv.best_params_,
                score=gscv.best_score_,
            )

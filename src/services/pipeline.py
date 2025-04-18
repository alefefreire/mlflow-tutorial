import logging
from typing import List, Tuple

import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.entities import Experiment
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from src.core.ml import BaseMLPipeline
from src.models.classifier import ClassifierModel
from src.models.data import Dataset
from src.models.params import Params
from src.models.regressor import RegressorModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.services.pipeline")


class MLPipeline(BaseMLPipeline):

    def __init__(self, dataset: Dataset, experiment: Experiment = None):
        self._dataset = dataset
        self._experiment = experiment

    def train_test_split(
        self,
        test_size: float = 0.2,
        is_stratified: bool = False,
        selected_features: List[str] = None,
        is_drop_id: bool = True,
        feature_Id: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into train and test sets.
        Parameters
        ----------
        test_size : float
            The proportion of the dataset to include in the test split.
        is_stratified : bool
            Whether to stratify the split based on the target variable.
        selected_features : List[str]
            The features to use for training.
        is_drop_id : bool
            Whether to drop the ID column from the dataset.
        feature_Id : List[str]
            The ID column to drop from the dataset.
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            The train and test sets.
        """
        logger.info("Splitting the dataset into train and test sets.")
        has_nan: bool = self._dataset.data.isna().any().any()
        if has_nan:
            self._dataset.data = self._dataset.data.dropna()

        X = self._dataset.data[self._dataset.features]
        y = self._dataset.data[self._dataset.target]

        if selected_features:
            X = X[selected_features]
        if is_drop_id and feature_Id:
            X = X.drop(feature_Id, axis=1)

        stratify = y if is_stratified else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

        train_dataset: PandasDataset = mlflow.data.from_pandas(
            pd.concat([X_train, y_train], axis=1), name="train_dataset", targets="Class"
        )
        test_dataset: PandasDataset = mlflow.data.from_pandas(
            pd.concat([X_test, y_test], axis=1), name="test_dataset", targets="Class"
        )
        with mlflow.start_run(
            experiment_id=self._experiment.experiment_id,
            run_name="train_test_split",
        ):
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="testing")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> ClassifierModel | RegressorModel:
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
        RegressorModel
            The trained model.
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
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        # Score the model
        score = f1_score(train_preds, y_train)
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
            scoring="f1_weighted",
        )
        bayes_search.fit(X_train, y_train)
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
        pass

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

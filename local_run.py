from typing import List, NamedTuple

from dotenv import load_dotenv
from mlflow.entities import Experiment
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier

from src.core.switcher import ModelSwitcher
from src.custom.dataprep import CustomDataPrep, CustomDataSplitter
from src.custom.trainer import CustomModelTrainer
from src.custom.tuner import CustomModelTuner
from src.models.params import Estimators, Params
from src.services.data_fetch import DataFetch
from src.services.pipeline import MLPipeline

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


class RunPipelineParams(NamedTuple):
    """
    Named tuple to hold the parameters for running the machine learning pipeline.

    Attributes
    ----------
    test_size : float
        Proportion of the dataset to include in the test split. Default is 0.2.
    is_stratified : bool
        Whether to perform stratified splitting of the dataset. Default is True.
    selected_features : List[str]
        List of feature names to be selected for training. Default is None.
    is_drop_id : bool
        Whether to drop the ID column from the dataset. Default is True.
    feature_Id : List[str]
        List of ID column names to be dropped. Default is None.
    n_splits : int
        Number of splits for cross-validation. Default is 5.
    random_state : int
        Random seed for reproducibility. Default is 42.
    n_iter : int
        Number of iterations for randomized search or similar processes. Default is 10.
    n_jobs : int
        Number of jobs to run in parallel. Default is -1 (use all processors).
    cv : int
        Number of cross-validation folds. Default is 5.
    verbose : int
        Verbosity level for logging. Default is 0.
    """

    test_size: float = 0.2
    is_stratified: bool = True
    selected_features: List[str] = None
    is_drop_id: bool = True
    feature_Id: List[str] = None
    n_splits: int = 5
    random_state: int = 42
    n_iter: int = 10
    n_jobs: int = -1
    cv: int = 5
    verbose: int = 0


class TrainParams(NamedTuple):
    """
    Named tuple to hold the parameters for the model training.

    Attributes
    ----------

    pre_processing : Params
        Parameters for the preprocessing step in the pipeline. Default is a PowerTransformer
        with the 'yeo-johnson' method and standardization enabled.
    estimator : ClfSwitcher
        The machine learning model or estimator to be used in the pipeline. Default is an
        XGBClassifier with 100 estimators, 'binary:logistic' objective, and random state set to 42.
    """

    pre_processing: Params = Params(
        method=PowerTransformer(
            method="yeo-johnson",
            standardize=True,
        )
    )
    estimators: Estimators = Estimators(
        xgboost=XGBClassifier(
            n_estimators=100,
            objective="binary:logistic",
            random_state=42,
        ),
        logistic_regression=LogisticRegression(max_iter=1000),
    )


class TunerParams(NamedTuple):
    """
    Named tuple to hold the parameters for model tuning.

    Attributes
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object containing preprocessing and model steps.
        Default is a pipeline with a PowerTransformer and a ModelSwitcher.
    param_grid : List[Params]
        A list of parameter grids to be used for hyperparameter tuning.
        Default includes parameter grids for XGBClassifier and LogisticRegression.
    scoring : str
        The scoring metric to be used for evaluating the models during tuning. Default is "f1_micro".
    enable_mlflow : bool
        Whether to enable MLflow logging during the tuning process. Default is False.
    cv : int
        Number of cross-validation folds. Default is 5.
    n_jobs : int
        Number of jobs to run in parallel. Default is -1 (use all processors).
    verbose : int
        Verbosity level for logging. Default is 0.
    """

    pipeline: Pipeline = Pipeline(
        steps=[
            (
                "pre_processing",
                PowerTransformer(method="yeo-johnson", standardize=True),
            ),
            ("model", ModelSwitcher()),
        ]
    )
    param_grid: List[Params] = [
        Params(
            model__estimator=[
                XGBClassifier(
                    n_estimators=100, objective="binary:logistic", random_state=42
                )
            ],
            model__estimator__max_depth=[3, 5, 7],
            model__estimator__min_child_weight=[1, 3, 5],
            model__estimator__subsample=[0.6, 0.8, 1.0],
        ),
        Params(
            model__estimator=[LogisticRegression(max_iter=1000)],
            model__estimator__C=[0.001, 0.01, 0.1],
        ),
    ]
    scoring: str = "f1_micro"
    enable_mlflow: bool = False
    cv: int = 5
    n_jobs: int = -1
    verbose: int = 0


def run(
    params: RunPipelineParams,
    train_params: TrainParams,
    tuner_params: TunerParams,
    experiment: Experiment = None,
) -> None:
    """
    Run the entire pipeline.
    """
    kaggle_api_client = KaggleApi()
    _ = kaggle_api_client.authenticate()

    data_fetch = DataFetch(kaggle_client=kaggle_api_client)
    dataset = data_fetch.fetch()

    data_prep = CustomDataPrep(dataset=dataset)
    data_splitter = CustomDataSplitter(experiment=experiment)
    model_trainer = CustomModelTrainer(
        experiment=experiment,
        pre_processing=train_params.pre_processing,
        estimators=train_params.estimators,
    )
    model_tuner = CustomModelTuner(
        tuner_params=tuner_params,
        experiment=experiment,
    )
    ml = MLPipeline(
        dataset=dataset,
        experiment=experiment,
        data_prep=data_prep,
        data_splitter=data_splitter,
        model_trainer=model_trainer,
        model_tuner=model_tuner,
    )
    ml.run_pipeline(
        selected_features=params.selected_features,
        is_drop_id=params.is_drop_id,
        feature_Id=params.feature_Id,
        test_size=params.test_size,
        is_stratified=params.is_stratified,
        n_splits=params.n_splits,
        random_state=params.random_state,
        n_iter=params.n_iter,
        n_jobs=params.n_jobs,
        cv=params.cv,
        verbose=params.verbose,
    )


if __name__ == "__main__":
    run(
        params=RunPipelineParams(),
        train_params=TrainParams(),
        tuner_params=TunerParams(),
    )
